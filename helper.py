import numpy as np
import torch
import torch.nn.functional as F
import time

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

from pathlib import Path

from tqdm import trange

from vggt.utils.pose_enc import pose_encoding_to_extri_intri

def align_extrinsic_seq(origin_position: np.ndarray, sequence: np.ndarray):
    assert origin_position.shape[0] == sequence.shape[1] == 3
    assert origin_position.shape[1] == sequence.shape[2] == 4
    
    extra_row = np.array([0, 0, 0, 1])
    origin_position = np.vstack([origin_position, extra_row])

    for i in range(len(sequence)):
        pos_16 =  np.vstack([sequence[i], extra_row])
        sequence[i] = (pos_16 @ origin_position)[:-1]
        
    return sequence

def parse_sequences_dict(model, images: torch.Tensor, dtype, resolution=518, sequence_length=20):
    extrinsic, intrinsic, depth_map, depth_conf = parse_sequences(model, images, dtype, resolution, sequence_length)
    
    predictions = { }
    predictions['extrinsic'] = extrinsic
    predictions['intrinsic'] = intrinsic
    predictions['depth'] = depth_map
    predictions['depth_conf'] = depth_conf
    predictions['images'] = images.cpu().detach().numpy()
    
    return predictions

def parse_sequences(model, images: torch.Tensor, dtype, resolution=518, sequence_length=20):
    assert sequence_length > 1
    sequences = list(images.split(sequence_length-1))
    # overlap last image in i_th sequence and first in next 
    # to be able to restore the shift between sequences
    for i in range(len(sequences)-1):
        sequences[i] = torch.stack([*sequences[i], sequences[i+1][0]]).cpu().detach()
    
    extrinsic_sequences = []
    intrinsic_sequences = []
    depth_map_sequences = []
    depth_conf_sequences = []
    
    images = images.cpu().detach()
    
    non_first_duration_sum = 0
    
    print(f"Sequence length: {sequence_length}")
    
    total_images_num = len(images) + len(sequences)-1
    with trange(total_images_num) as t:
        for i, sequence in enumerate(sequences):
            sequence = sequence.to('cuda:0')
            start_t = time.time()
            extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, sequence, dtype, resolution)
            
            start_idx = int(i > 0)
            if i > 0:
                extrinsic = align_extrinsic_seq(extrinsic_sequences[-1][-1], extrinsic)
                
            extrinsic_sequences.append(extrinsic[start_idx:])
            intrinsic_sequences.append(intrinsic[start_idx:])
            depth_map_sequences.append(depth_map[start_idx:])
            depth_conf_sequences.append(depth_conf[start_idx:])
            
            sequence = sequence.to('cpu')
            
            torch.cuda.empty_cache()

            t.update(len(sequence))
            end_t = time.time()
            duration = end_t-start_t
            
            if i == 0:
                print(f"First iteration time: {duration / len(sequence)} s/frame")
            else:
                non_first_duration_sum += duration
                
    
    non_first_iter_time = non_first_duration_sum / (total_images_num-sequence_length) if total_images_num > sequence_length else 0
    print(f"Avg non-first iteration time: {non_first_iter_time:.3} s/frame")
                
        
    extrinsic = np.concatenate(extrinsic_sequences)
    intrinsic = np.concatenate(intrinsic_sequences)
    depth_map = np.concatenate(depth_map_sequences)
    depth_conf = np.concatenate(depth_conf_sequences)
    
    return extrinsic, intrinsic, depth_map, depth_conf

def run_VGGT(model, images: torch.Tensor, dtype, resolution=518):
    # images: [B, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)
        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        torch.cuda.empty_cache()
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        torch.cuda.empty_cache()
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
        torch.cuda.empty_cache()

    extrinsic = extrinsic.squeeze(0).cpu().detach().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().detach().numpy()
    depth_map = depth_map.squeeze(0).cpu().detach().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().detach().numpy()
    
    return extrinsic, intrinsic, depth_map, depth_conf