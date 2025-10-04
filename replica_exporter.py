from typing import List, Dict, Tuple
from pathlib import Path
import os
import numpy as np

import torch
import torchvision.transforms.v2 as transforms
import cv2 as cv

import pycolmap
import yaml

def tum_to_kitti(tx, ty, tz, qw, qx, qy, qz):
    # https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
    
    r00 = 2*(qw**2 + qx**2) - 1
    r01 = 2*(qx * qy - qw * qz)
    r02 = 2*(qx * qz + qw * qy)
    r10 = 2*(qx * qy + qw * qz)
    r11 = 2*(qw**2 + qy**2) - 1
    r12 = 2*(qy * qz - qw * qx)
    r20 = 2*(qx * qz - qw * qy)
    r21 = 2*(qy * qz + qw * qx)
    r22 = 2*(qw**2 + qz**2) - 1
    R = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
    
    # add position shift
    t = np.array([[tx], [ty], [tz]])
    H = np.hstack((R, t))
    H = np.vstack((H, [0, 0, 0, 1]))
    
    # get all 16 values including last row 0 0 0 1
    return H.flatten()

def make_str_num(num, digits=4):
    str_num = str(num)
    zeros_num = digits-len(str_num)
    zeros = '0' * (zeros_num if zeros_num > 0 else 0)
    return zeros + str_num

class ReplicaExporter:
    def __init__(self, 
        dataset_root_dir: str
    ):
        self.root_dir = Path(dataset_root_dir)
        
    def export(self,
        orig_coords: torch.Tensor,
        frames: torch.Tensor,
        depth_maps: torch.Tensor,
        reconstruction:  pycolmap.Reconstruction,
        scene_name:str=None
    ):
        assert len(orig_coords) == len(frames) == len(depth_maps)
        
        if not scene_name:
            scene_name = f'scene_{make_str_num(len(os.listdir(self.root_dir)), 2)}'
            
        self._init_paths(scene_name)
        self.orig_coords = orig_coords.to(torch.int32)
        
        self._save_processed_frames(frames)
        self._save_depth_maps(depth_maps)
        self._save_colmap_as_kitti_traj(reconstruction)
        self._save_colmap_as_camera_yaml(reconstruction)
        

    def _init_paths(self, scene_name: str):
        self.scene_dir = self.root_dir / scene_name
        self.results_dir = self.scene_dir / 'results'
        self.traj_path = self.scene_dir / 'traj.txt'
        self.camera_path = self.scene_dir / 'camera.yaml'
        self.last_camera_path = self.root_dir / 'camera.yaml'
        
        self.frame_stem = 'frame'
        self.frame_suffix_wo_dot = 'jpg'
        
        self.depth_stem = 'depth'
        self.depth_suffix_wo_dot = 'png'
        
        for dir in [self.scene_dir, self.results_dir]:
            os.makedirs(dir, exist_ok=True)
        
    def _make_result_path(self, stem: str, frame_num: int, suffix_without_dot: str):
        return self.results_dir / f'{stem}_{make_str_num(frame_num)}.{suffix_without_dot}'
    
    def _make_frame_path(self, frame_num: int):
        return self._make_result_path(self.frame_stem, frame_num, self.frame_suffix_wo_dot)
    
    def _make_depth_path(self, frame_num: int):
        return self._make_result_path(self.depth_stem, frame_num, self.depth_suffix_wo_dot)
        
        
    def _restore_tensor_with_coords(self, tensor: torch.Tensor, coords: torch.Tensor):
        '''
        `tensor`: shape must be `[... , C, H, W]` or `[H, W]`
        '''
        assert len(tensor.shape) > 1
        
        x1, y1, x2, y2, w, h = coords
        
        tensor_whc = tensor.permute(list(range(len(tensor.shape)))[::-1])
        cropped = tensor_whc[x1:x2, y1:y2]
        tensor_chw = cropped.permute(list(range(len(cropped.shape)))[::-1])
        
        resize = transforms.Resize([h, w])
        tensor_chw = tensor_chw if len(tensor_chw.shape) > 2 else tensor_chw.unsqueeze(0)
        resized = resize(tensor_chw)
        
        return resized if len(tensor.shape) > 2 else resized.squeeze(0)
          
    def _save_processed_frames(self, frames):
        _frames = frames.squeeze()
        _frames *= 255
        _frames = _frames.clamp(0.0, 255)
        
        for i, (frame, coords) in enumerate(zip(_frames, self.orig_coords)):
            restored = self._restore_tensor_with_coords(frame, coords)
            restored = restored.permute(1, 2, 0) # CHW -> HWC
            restored = restored.cpu().detach().numpy().astype(np.uint8)
            rgb_to_brg = [2, 1, 0]
            img_cv = restored[:, :, rgb_to_brg]
            cv.imwrite(self._make_frame_path(i), img_cv)
            
    def _save_depth_maps(self, depth_maps: torch.Tensor):
        _depth_maps = depth_maps.squeeze()
        self.depth_scale = _depth_maps.max()
        _depth_maps /= self.depth_scale
        _depth_maps *= 255
        _depth_maps = _depth_maps.clamp(0.0, 255)
        
        print(f'Scale is {self.depth_scale}')
        
        for i, (depth_map, coords) in enumerate(zip(_depth_maps, self.orig_coords)):
            restored = self._restore_tensor_with_coords(depth_map, coords)
            img_cv = restored.cpu().detach().numpy().astype(np.uint8)
            cv.imwrite(self._make_depth_path(i), img_cv)
            
    def _save_colmap_as_kitti_traj(self, reconstruction: pycolmap.Reconstruction):
        images = list(reconstruction.images.values())
        images.sort(key=lambda x: x.name)
        
        with open(self.traj_path, 'w') as f:
            with open(self.traj_path.with_stem('traj_cam_from_world'), 'w') as f_cfw:
                for image in images:
                    # camera-from-world
                    homogen_mat_12 = image.cam_from_world.matrix()
                    homogen_mat_16_cfw = np.vstack([homogen_mat_12, [0, 0, 0, 1]])
                    # world-from-camera
                    homogen_mat_16_wfc = np.linalg.inv(homogen_mat_16_cfw)
                    
                    kitti_str = ' '.join(map(str, homogen_mat_16_wfc.flatten()))
                    f.write(kitti_str + '\n')
                    
                    kitti_str = ' '.join(map(str, homogen_mat_16_cfw.flatten()))
                    f_cfw.write(kitti_str + '\n')
    
                        
    def _save_colmap_as_camera_yaml(self, reconstruction: pycolmap.Reconstruction):
        cameras = list(reconstruction.cameras.values())
        assert len(cameras) > 0
        base_cam = cameras[0]
        
        image_heights = np.array([camera.width for camera in cameras])
        image_widths = np.array([camera.height for camera in cameras])
        fxs = np.array([camera.focal_length_x for camera in cameras])
        fys = np.array([camera.focal_length_y for camera in cameras])
        cxs = np.array([camera.principal_point_x for camera in cameras])
        cys = np.array([camera.principal_point_y for camera in cameras])
        
        assert all(image_heights == image_heights[0])
        assert all(image_widths == image_widths[0])
        
        camera_dict = {
            'dataset_name': 'replica',
            'camera_params': {
                'image_height': int(image_heights[0]),
                'image_width': int(image_widths[0]),
                'fx': float(fxs.mean()),
                'fy': float(fys.mean()),
                'cx': int(cxs.mean()),
                'cy': int(cys.mean()),
                'crop_edge': 0,
                'png_depth_scale': float(self.depth_scale),
            }
        }
        
        with open(self.camera_path, 'w') as f:
            yaml.dump_all([camera_dict], f)
            
        with open(self.last_camera_path, 'w') as f:
            yaml.dump_all([camera_dict], f)
        