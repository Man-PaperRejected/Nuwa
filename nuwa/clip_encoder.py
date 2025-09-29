#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# Copyright 2024 Senqiao Yang
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPAttention, CLIPEncoder
from .utils import CLIPAttention_forward, CLIP_EncoderLayer_forward
import os
import math
import pdb
import re

max_dist = math.sqrt(1058)
def _create_distance_penalty_matrix(
    grid_size=(24, 24), 
    distance_threshold=math.sqrt(280),  
    device='cpu'
):
    """
    Creates a matrix where each element (i, j) is a penalty based on the
    2D grid distance between token i and token j.
    
    Penalty logic:
    - If distance <= threshold, Penalty = 1 - (distance / threshold)
    - If distance > threshold, Penalty = 0
    """
    H, W = grid_size
    coords = torch.stack(
        torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij'), 
        dim=-1
    ).float()
    
    coords_flat = coords.view(-1, 2)
    
    dist_matrix = torch.cdist(coords_flat, coords_flat, p=2.0)

    normalized_dist = dist_matrix / distance_threshold
    

    clipped_dist = torch.clamp(normalized_dist, max=1.0)

    dist_penalty = 1.0 - clipped_dist
    
    return dist_penalty

def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor

def _stitch_features(features, num_patches_height, num_patches_width,image_sizes):
    patch_grid_per_image = 24
    is_3d = features.dim() == 3
    if is_3d:
        N, num_tokens, D = features.shape
    else:
        N, num_tokens = features.shape; D = 1; features = features.unsqueeze(-1)
    features_grid = features.view(N, patch_grid_per_image, patch_grid_per_image, D)
    features_grid = features_grid.view(num_patches_height, num_patches_width, patch_grid_per_image, patch_grid_per_image, D)
    features_grid = features_grid.permute(4, 0, 2, 1, 3).contiguous()
    features_grid = features_grid.flatten(1, 2).flatten(2, 3)
    features_grid = unpad_image(features_grid,image_sizes)
    unpad_size = features_grid.shape[1:]
    features_grid = features_grid.flatten(1, 2).transpose(0, 1)
    
    stitched_features = features_grid.reshape(1, -1, D)
    if not is_3d:
        stitched_features = stitched_features.squeeze(-1)
    return stitched_features,unpad_size

def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        # Calculate the downscaled size to keep the aspect ratio
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)

        # Calculate effective and wasted resolutions
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit

def get_anyres_image_grid_shape(image_size, grid_pinpoints = [[336,672],[672,336],[672,672],[1008,336],[336,1008]], patch_size=336):
    
    grid_pinpoints = [[336,672],[672,336],[672,672],[1008,336],[336,1008]]
    
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
        # Multiply all elements by patch_size
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints

    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size

def create_instance_dir(parent_dir="."):
    os.makedirs(parent_dir, exist_ok=True)
    i = max([int(d.split('_')[-1]) for d in os.listdir(parent_dir) 
             if d.startswith('instance_') and d.split('_')[-1].isdigit()], default=-1) + 1
    new_dir = os.path.join(parent_dir, f"instance_{i}")
    os.makedirs(new_dir)
    return new_dir

class CLIPVisionTower_Nuwa(nn.Module):
    
    @torch.no_grad()
    def forward(self, images):
        # PENALTY_THRESHOLD_PERCENTILE =  float(os.getenv("THRE"))
        PENALTY_THRESHOLD_PERCENTILE = 0.55
        TOKEN_CONFIGS = {
        64: {'top_n_per_region': 1},
        128: {'top_n_per_region': 2}, 
        192: {'top_n_per_region': 3},
        160: {'top_n_per_region': 1},
        320: {'top_n_per_region': 3},
        640: {'top_n_per_region': 4},
        }
        Stage1_CONFIGS={
            64:112,
            128:224,
            192:336,
            160:290,
            320:582,
            640:1162,
        }
        
        matain_token =  self.vision_tower._info["matain_token"]
        distance = self.vision_tower._info["distance"]
        
        config = TOKEN_CONFIGS[matain_token]
        num_tokens = Stage1_CONFIGS[matain_token]
        B = images.shape[0] # Get batch size
        total_tokens_to_keep = num_tokens
        top_n_per_region = config['top_n_per_region']

        num_tokens_per_image = min(math.ceil(total_tokens_to_keep / B), 144*top_n_per_region)
        
        
        
        
        if not hasattr(self, 'dist_penalty'):
            patch_grid_size = (24, 24)
            dist_penalty = _create_distance_penalty_matrix(patch_grid_size,distance_threshold=math.sqrt(distance), device=self.device)
            self.register_buffer('dist_penalty', dist_penalty, persistent=False)
            self.patch_grid_size = patch_grid_size



        image_forward_outs = self.vision_tower(
            images.to(device=self.vision_tower.device, dtype=self.vision_tower.dtype), 
            output_hidden_states=True, 
            output_attentions=True
        )
        
        all_hidden_states = image_forward_outs.hidden_states
        hidden_states_for_aggregation = all_hidden_states[self.select_layer]
        attentions = image_forward_outs.attentions[self.select_layer]
        _B, N, D = hidden_states_for_aggregation.shape
        num_patches = N - 1
        H, W = self.patch_grid_size
        
        if num_patches != H * W:
            raise ValueError(f"Expected {H*W} patches, but got {num_patches}.")

        # --- Calculate Metric Map  ---

        cls_attention = attentions[:, :, 0, 1:] 
        
        metric_map = cls_attention.sum(dim=1)

        
        # --- 1. MODIFIED: Select Top-N Candidates from 2x2 Grids ---
        metric_2d = metric_map.view(B, H, W)
        
        
        
        
        
        unfolded_attn = metric_2d.unfold(1, 2, 2).unfold(2, 2, 2)
        num_regions = (H // 2) * (W // 2) # 144 regions
        region_size = 2 * 2 # 4 tokens per region
        regions = unfolded_attn.reshape(B, num_regions, region_size)
        
        # MODIFIED: Instead of argmax (top-1), use topk to get top_n_per_region
        # This gives us the indices of the best N tokens *within each 2x2 region*.
        _, local_topk_indices = regions.topk(k=top_n_per_region, dim=-1) # Shape: (B, 144, top_n)
        
        # Convert local (0-3) indices to local 2D coordinates (y, x)
        local_y = local_topk_indices // 2
        local_x = local_topk_indices % 2
        
        # Get global region coordinates
        region_ids = torch.arange(num_regions, device=images.device).unsqueeze(0)
        regions_per_row = W // 2
        region_y = region_ids // regions_per_row
        region_x = region_ids % regions_per_row
        

        global_y = region_y.unsqueeze(-1) * 2 + local_y
        global_x = region_x.unsqueeze(-1) * 2 + local_x
        
        # Calculate the 1D indices for all candidates
        candidate_indices = global_y * W + global_x 
        
        candidate_indices = candidate_indices.view(B, -1)
        candidate_scores = torch.gather(metric_map, 1, candidate_indices)

        _, top_candidate_indices = torch.topk(candidate_scores, num_tokens_per_image, dim=1)

        benchmark_indices = torch.gather(candidate_indices, 1, top_candidate_indices)
        
        # Sort indices for potential positional embedding requirements downstream
        benchmark_indices = benchmark_indices.sort().values
        
        # --- 3. Calculate Similarity Graph ---
        if len(all_hidden_states) < 25:
            raise ValueError("Model must have at least 24 layers to average from layers 16-24.")
        
        stacked_hs = torch.stack(all_hidden_states[16:25], dim=0)
        avg_hidden_states_for_sim = stacked_hs.mean(dim=0)
        
        patch_tokens_for_sim = avg_hidden_states_for_sim[:, 1:, :]
        patch_tokens_norm = F.normalize(patch_tokens_for_sim, p=2, dim=-1)
        sim_matrix = torch.bmm(patch_tokens_norm, patch_tokens_norm.transpose(1, 2))
        
        aggregation_weights = F.relu(sim_matrix) * self.dist_penalty
        # --- NEW: Apply Selective Penalty Based on Benchmark Tokens ---
        benchmark_scores = torch.gather(metric_map, 1, benchmark_indices)
        score_threshold = torch.quantile(benchmark_scores.to(torch.float), PENALTY_THRESHOLD_PERCENTILE, dim=1, keepdim=True)
        is_high_attention_token = benchmark_scores >= score_threshold
        
        selective_penalty_matrix = torch.ones(B, num_tokens_per_image, num_patches, device=images.device, dtype=aggregation_weights.dtype)
        selective_penalty_matrix[is_high_attention_token] = 0
        selective_penalty_matrix.scatter_(
            dim=2, 
            index=benchmark_indices.unsqueeze(-1), 
            value=1.0 
            )
        
        aggregation_weights = F.relu(sim_matrix) * self.dist_penalty
        # --- 4. Aggregate Information to the Final Benchmark Tokens  ---
        patch_tokens_for_aggregation = hidden_states_for_aggregation[:, 1:, :]
        
        benchmark_indices_expanded = benchmark_indices.unsqueeze(-1).expand(-1, -1, num_patches)
        benchmark_weights = torch.gather(aggregation_weights, 1, benchmark_indices_expanded)
        # pdb.set_trace()
        benchmark_weights = benchmark_weights * selective_penalty_matrix
        
        benchmark_weights_norm = benchmark_weights / (benchmark_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        self_indices = benchmark_indices.unsqueeze(-1)
        benchmark_weights_norm.scatter_(dim=2, index=self_indices, value=1.0)
        
        aggregated_tokens = torch.bmm(benchmark_weights_norm.to(patch_tokens_for_aggregation.dtype), patch_tokens_for_aggregation)
        
        # offsets = torch.arange(B, device=benchmark_indices.device) * 576
        # offsets = offsets.unsqueeze(1)
        # benchmark_indices += offsets
        # instance = {"indices":benchmark_indices[0].cpu()}
        # torch.save(instance, os.path.join(base_dir, f"indices.pt"))
        return aggregated_tokens.to(images.dtype), benchmark_indices.squeeze(0)
    
def prune_vision_tokens(
    hidden_states_for_aggregation: torch.Tensor,
    hidden_states_for_sim: torch.Tensor,
    cls_attention_map: torch.Tensor,
    patch_grid_size: tuple[int, int],
    num_tokens_to_keep: int,
    distance_threshold: float,
    top_n_per_region: int = 1,
    penalty_threshold_percentile: float = 0.55
):

    B, num_patches, D = hidden_states_for_aggregation.shape
    H, W = patch_grid_size
    device = hidden_states_for_aggregation.device

    if num_tokens_to_keep > top_n_per_region*H * W/4 :
        top_n_per_region = int(math.ceil((num_tokens_to_keep*4) / (H*W)))
        if top_n_per_region >=4:
            top_n_per_region=4
            num_tokens_to_keep =  H * W


    metric_map = cls_attention_map
    metric_2d = metric_map.view(B, H, W)
    unfolded_attn = metric_2d.unfold(1, 2, 2).unfold(2, 2, 2)
    num_regions_h, num_regions_w = H // 2, W // 2
    num_regions = num_regions_h * num_regions_w
    regions = unfolded_attn.reshape(B, num_regions, 4)
    _, local_topk_indices = regions.topk(k=top_n_per_region, dim=-1)
    local_y, local_x = local_topk_indices // 2, local_topk_indices % 2
    region_ids = torch.arange(num_regions, device=device).unsqueeze(0)
    region_y, region_x = region_ids // num_regions_w, region_ids % num_regions_w
    global_y = region_y.unsqueeze(-1) * 2 + local_y
    global_x = region_x.unsqueeze(-1) * 2 + local_x
    candidate_indices = (global_y * W + global_x).view(B, -1)
    
    candidate_scores = torch.gather(metric_map, 1, candidate_indices)
    
    _, top_candidate_indices = torch.topk(candidate_scores, num_tokens_to_keep, dim=1)
    benchmark_indices = torch.gather(candidate_indices, 1, top_candidate_indices)
    benchmark_indices = benchmark_indices.sort(dim=1).values

    dist_penalty = _create_distance_penalty_matrix(patch_grid_size, distance_threshold, device)
    patch_tokens_norm = F.normalize(hidden_states_for_sim, p=2, dim=-1)
    sim_matrix = torch.bmm(patch_tokens_norm, patch_tokens_norm.transpose(1, 2))
    aggregation_weights = F.relu(sim_matrix) * dist_penalty

    benchmark_scores = torch.gather(metric_map, 1, benchmark_indices)
    score_threshold = torch.quantile(benchmark_scores.float(), penalty_threshold_percentile, dim=1, keepdim=True)
    is_high_attention_token = benchmark_scores >= score_threshold

    selective_penalty_matrix = torch.ones(B, num_tokens_to_keep, num_patches, device=device, dtype=aggregation_weights.dtype)
    

    selective_penalty_matrix[is_high_attention_token] = 0.0

    selective_penalty_matrix.scatter_(dim=2, index=benchmark_indices.unsqueeze(2), value=1.0)

    benchmark_indices_expanded = benchmark_indices.unsqueeze(-1).expand(-1, -1, num_patches)
    benchmark_weights = torch.gather(aggregation_weights, 1, benchmark_indices_expanded)
    
    benchmark_weights = benchmark_weights * selective_penalty_matrix
    
    benchmark_weights_norm = benchmark_weights / (benchmark_weights.sum(dim=-1, keepdim=True) + 1e-8)
    benchmark_weights_norm.scatter_(dim=2, index=benchmark_indices.unsqueeze(-1), value=1.0) 
    
    aggregated_tokens = torch.bmm(benchmark_weights_norm.to(hidden_states_for_aggregation.dtype), hidden_states_for_aggregation)
    
    return aggregated_tokens, benchmark_indices


class CLIPVisionTower_Nuwa_Next(nn.Module):
    
    @torch.no_grad()
    def forward(self, images, image_sizes):
        PENALTY_THRESHOLD_PERCENTILE =  float(os.getenv("THRE"))
        # PENALTY_THRESHOLD_PERCENTILE = 0.55
        TOKEN_CONFIGS = {
        64:1,
        128: 2, 
        192: 3,
        160: 2,
        320: 3,
        640: 3,
        }
        Stage1_CONFIGS={
            64:112,
            128:224,
            192:336,
            160:290,
            320:582,
            640:1162,
        }
        B,_,_,_,= images.shape
        
        matain_token =  self.vision_tower._info["matain_token"]
        distance = self.vision_tower._info["distance"]
        token_per_image = Stage1_CONFIGS[matain_token] // B
        top_n_per_region = TOKEN_CONFIGS[matain_token]


        image_forward_outs = self.vision_tower(
            images.to(device=self.vision_tower.device, dtype=self.vision_tower.dtype), 
            output_hidden_states=True, 
            output_attentions=True
        )
        
        all_hidden_states = image_forward_outs.hidden_states
        hidden_states_for_aggregation = all_hidden_states[self.select_layer][:, 1:, :]
        
        stacked_hs = torch.stack(all_hidden_states[16:25], dim=0)
        hidden_states_for_sim = stacked_hs.mean(dim=0)[:, 1:, :]
        
        attentions = image_forward_outs.attentions[self.select_layer]
        cls_attention_map = attentions[:, :, 0, 1:].sum(dim=1)
        
        global_hs_agg = hidden_states_for_aggregation[0:1]
        global_hs_sim = hidden_states_for_sim[0:1]
        global_attn_map = cls_attention_map[0:1]
        
        global_tokens, global_indices = prune_vision_tokens(
            hidden_states_for_aggregation=global_hs_agg,
            hidden_states_for_sim=global_hs_sim,
            cls_attention_map=global_attn_map,
            patch_grid_size=(24, 24),
            num_tokens_to_keep=token_per_image,
            distance_threshold=math.sqrt(distance),
            top_n_per_region=top_n_per_region,
            penalty_threshold_percentile=PENALTY_THRESHOLD_PERCENTILE
        )
        
        if images.shape[0] > 1:
            local_hs_agg = hidden_states_for_aggregation[1:]
            local_hs_sim = hidden_states_for_sim[1:]
            local_attn_map = cls_attention_map[1:]
            
            num_patches_height, num_patches_width = get_anyres_image_grid_shape(image_sizes[0])
            num_local_images = local_hs_agg.shape[0]

            stitched_hs_agg,unpad_size = _stitch_features(local_hs_agg, num_patches_height, num_patches_width,image_sizes[0])
            stitched_hs_sim,_ = _stitch_features(local_hs_sim, num_patches_height, num_patches_width,image_sizes[0])
            stitched_attn_map,_ = _stitch_features(local_attn_map, num_patches_height, num_patches_width,image_sizes[0])
            
            
            num_tokens_local = token_per_image * num_local_images 

            local_tokens, local_indices = prune_vision_tokens(
                hidden_states_for_aggregation=stitched_hs_agg,
                hidden_states_for_sim=stitched_hs_sim,
                cls_attention_map=stitched_attn_map,
                patch_grid_size=unpad_size,
                num_tokens_to_keep=num_tokens_local,
                distance_threshold=math.sqrt(distance),
                top_n_per_region=top_n_per_region,
                penalty_threshold_percentile=PENALTY_THRESHOLD_PERCENTILE
            )
            num_global_patches = 24 * 24
            local_indices_offset = local_indices + num_global_patches
            
            final_tokens = torch.cat([global_tokens, local_tokens], dim=1)
            final_indices = torch.cat([global_indices.squeeze(0), local_indices_offset.squeeze(0)], dim=0)
            

        
        return final_tokens.to(images.dtype), final_indices,unpad_size
    
    
    
    
class CLIPVisionTower_Nuwa_abli(nn.Module):
    
    @torch.no_grad()
    def forward(self, images):
        # ==================== 消融实验控制变量 ====================
        # 1. 是否分区域提取基准token (默认: True)
        #    设置为 'False' 时，将从全局注意力图中直接选择top-k个token
        REGION = os.getenv('REGION', 'True').upper() == 'TRUE'
        
        # 2. 是否通过L2范数区分基准token并阻止其聚合 (默认: False)
        #    设置为 'True' 时，高L2范数的基准token将不参与特征聚合
        #    设置为 'False' 时，所有基准token都参与特征聚合
        L2NORM = os.getenv('L2NORM', 'False').upper() == 'TRUE'
        
        # print(f"Ablation settings: REGION={REGION}, L2NORM={L2NORM}")
        # ==========================================================

        PENALTY_THRESHOLD_PERCENTILE = float(os.getenv("THRE", "0.55"))
        
        TOKEN_CONFIGS = {
            64: {'top_n_per_region': 1},
            128: {'top_n_per_region': 2}, 
            192: {'top_n_per_region': 3},
            160: {'top_n_per_region': 1},
            320: {'top_n_per_region': 3},
            640: {'top_n_per_region': 4},
        }
        Stage1_CONFIGS = {
            64: 112,
            128: 224,
            192: 336,
            160: 290,
            320: 582,
            640: 1162,
        }
        
        matain_token = self.vision_tower._info["matain_token"]
        distance = self.vision_tower._info["distance"]
        
        config = TOKEN_CONFIGS[matain_token]
        num_tokens = Stage1_CONFIGS[matain_token]
        B = images.shape[0]
        total_tokens_to_keep = num_tokens
        top_n_per_region = config['top_n_per_region']
        # pdb.set_trace()
        max_tokens_regional = 144 * top_n_per_region
        num_tokens_per_image = min(math.ceil(total_tokens_to_keep / B), max_tokens_regional if REGION else 576)

        if not hasattr(self, 'dist_penalty'):
            patch_grid_size = (24, 24)
            dist_penalty = _create_distance_penalty_matrix(patch_grid_size, distance_threshold=math.sqrt(distance), device=self.device)
            self.register_buffer('dist_penalty', dist_penalty, persistent=False)
            self.patch_grid_size = patch_grid_size

        image_forward_outs = self.vision_tower(
            images.to(device=self.vision_tower.device, dtype=self.vision_tower.dtype), 
            output_hidden_states=True, 
            output_attentions=True
        )
        
        all_hidden_states = image_forward_outs.hidden_states
        hidden_states_for_aggregation = all_hidden_states[self.select_layer]
        attentions = image_forward_outs.attentions[self.select_layer]
        _B, N, D = hidden_states_for_aggregation.shape
        num_patches = N - 1
        H, W = self.patch_grid_size
        
        if num_patches != H * W:
            raise ValueError(f"Expected {H*W} patches, but got {num_patches}.")

        cls_attention = attentions[:, :, 0, 1:] 
        metric_map = cls_attention.sum(dim=1)

        if REGION:
            metric_2d = metric_map.view(B, H, W)
            unfolded_attn = metric_2d.unfold(1, 2, 2).unfold(2, 2, 2)
            num_regions = (H // 2) * (W // 2)
            region_size = 2 * 2
            regions = unfolded_attn.reshape(B, num_regions, region_size)
            
            _, local_topk_indices = regions.topk(k=top_n_per_region, dim=-1)
            
            local_y, local_x = local_topk_indices // 2, local_topk_indices % 2
            
            region_ids = torch.arange(num_regions, device=images.device).unsqueeze(0)
            regions_per_row = W // 2
            region_y, region_x = region_ids // regions_per_row, region_ids % regions_per_row
            
            global_y = region_y.unsqueeze(-1) * 2 + local_y
            global_x = region_x.unsqueeze(-1) * 2 + local_x
            
            candidate_indices = (global_y * W + global_x).view(B, -1)
            candidate_scores = torch.gather(metric_map, 1, candidate_indices)

            _, top_candidate_indices = torch.topk(candidate_scores, num_tokens_per_image, dim=1)
            benchmark_indices = torch.gather(candidate_indices, 1, top_candidate_indices)
        else:
            _, benchmark_indices = torch.topk(metric_map, k=num_tokens_per_image, dim=1)

        benchmark_indices = benchmark_indices.sort().values
        
        # --- 3. 计算相似度图 ---
        if len(all_hidden_states) < 25:
            raise ValueError("Model must have at least 24 layers to average from layers 16-24.")
        
        stacked_hs = torch.stack(all_hidden_states[16:25], dim=0)
        avg_hidden_states_for_sim = stacked_hs.mean(dim=0)
        
        patch_tokens_for_sim = avg_hidden_states_for_sim[:, 1:, :]
        patch_tokens_norm = F.normalize(patch_tokens_for_sim, p=2, dim=-1)
        sim_matrix = torch.bmm(patch_tokens_norm, patch_tokens_norm.transpose(1, 2))
        
        aggregation_weights = F.relu(sim_matrix) * self.dist_penalty
        
        # --- 4. 聚合信息到最终的基准Token ---
        patch_tokens_for_aggregation = hidden_states_for_aggregation[:, 1:, :]
        
        benchmark_indices_expanded = benchmark_indices.unsqueeze(-1).expand(-1, -1, num_patches)
        benchmark_weights = torch.gather(aggregation_weights, 1, benchmark_indices_expanded)
        
        benchmark_weights_norm = benchmark_weights / (benchmark_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        self_indices = benchmark_indices.unsqueeze(-1)
        benchmark_weights_norm.scatter_(dim=2, index=self_indices, value=1.0)
        
        # 首先，计算所有基准token的聚合后特征
        aggregated_tokens_all = torch.bmm(benchmark_weights_norm.to(patch_tokens_for_aggregation.dtype), patch_tokens_for_aggregation)
        
        # ==================== 消融点 2: L2范数选择性聚合 ====================
        if L2NORM:
            # 如果启用此机制，则找出高L2范数的token，并用它们聚合前的原始特征替换掉聚合后的结果
            
            # 1. 计算所有patch token的L2范数
            token_l2_norms = torch.linalg.norm(patch_tokens_for_aggregation, ord=2, dim=-1)
            # 2. 提取出基准token对应的L2范数
            benchmark_l2_norms = torch.gather(token_l2_norms, 1, benchmark_indices)
            
            # 3. 根据百分位阈值，确定哪些是“高范数”token
            norm_threshold = torch.quantile(benchmark_l2_norms.to(torch.float), PENALTY_THRESHOLD_PERCENTILE, dim=1, keepdim=True)
            is_high_norm_token = benchmark_l2_norms >= norm_threshold # Shape: (B, num_tokens_per_image)

            # 4. 获取基准token的原始特征向量
            benchmark_indices_expanded_for_features = benchmark_indices.unsqueeze(-1).expand(-1, -1, D)
            original_benchmark_features = torch.gather(patch_tokens_for_aggregation, 1, benchmark_indices_expanded_for_features)

            # 5. 使用torch.where，根据is_high_norm_token掩码进行选择
            # 如果是高范数token (mask=True)，则使用其原始特征
            # 否则 (mask=False)，使用其聚合后的特征
            mask_expanded = is_high_norm_token.unsqueeze(-1).expand_as(aggregated_tokens_all)
            final_aggregated_tokens = torch.where(mask_expanded, original_benchmark_features, aggregated_tokens_all)

        else:
            # 如果不启用此机制，则所有基准token都使用聚合后的特征
            final_aggregated_tokens = aggregated_tokens_all
        # ====================================================================
        
        return final_aggregated_tokens.to(images.dtype), benchmark_indices.squeeze(0)