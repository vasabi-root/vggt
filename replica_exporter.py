from typing import List, Dict, Tuple
from pathlib import Path
import os
import numpy as np

import torch
import torchvision.transforms.v2 as transforms
import cv2 as cv

def make_str_num(num, digits=4):
    str_num = str(num)
    zeros_num = digits-len(str_num)
    zeros = '0' * (zeros_num if zeros_num > 0 else 0)
    return zeros + str_num

class ReplicaExporter:
    def __init__(self, 
        export_dir: str
    ):
        self._init_paths(export_dir)
        
    def export(self,
        orig_coords: torch.Tensor,
        frames: torch.Tensor,
        depth_maps: torch.Tensor,
        export_dir:str=None
    ):
        assert len(orig_coords) == len(frames) == len(depth_maps)
        if export_dir:
            self._init_paths(export_dir)
        self.orig_coords = orig_coords.to(torch.int32)
        
        self._save_processed_frames(frames)
        self._save_depth_maps(depth_maps)

    def _init_paths(self, export_dir: str):
        self.dir = Path(export_dir)
        self.results_dir = self.dir / 'results'
        self.traj_path = self.dir / 'traj.txt'
        
        self.frame_stem = 'frame'
        self.frame_suffix_wo_dot = 'jpg'
        
        self.depth_stem = 'depth'
        self.depth_suffix_wo_dot = 'png'
        
        for dir in [self.dir, self.results_dir]:
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