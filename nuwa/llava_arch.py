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

from abc import ABC, abstractmethod
import copy
import torch
import torch.nn as nn

import pdb
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
# from llava.utils import rank0_print
import torch.nn.functional as F


def position_rebuild(keep_idxs,num_patch_width,num_patch_height,patch_side_len = 24):
    K = keep_idxs.shape[0]
    patch_ids = torch.arange(K, device=keep_idxs.device)
    patch_rows = patch_ids // num_patch_width
    patch_cols = patch_ids % num_patch_width
    
    local_token_rows = keep_idxs // patch_side_len
    local_token_cols = keep_idxs % patch_side_len
    global_token_rows = patch_rows.unsqueeze(1) * patch_side_len + local_token_rows
    global_token_cols = patch_cols.unsqueeze(1) * patch_side_len + local_token_cols
    global_width_in_tokens = num_patch_width * patch_side_len
    global_keep_idxs = global_token_rows * global_width_in_tokens + global_token_cols

    return global_keep_idxs

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

def insert_newlines_after_pruning(
    pruned_tokens: torch.Tensor,
    pruned_indices: torch.Tensor,
    unpadded_grid_size: tuple[int, int],
    newline_token_embedding: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    在裁剪后的token序列中，根据其原始空间位置插入换行符。

    Args:
        pruned_tokens (torch.Tensor): 裁剪后的token特征 (1, K_pruned, D)。
        pruned_indices (torch.Tensor): 裁剪后token在unpad网格中的索引 (1, K_pruned)。
        unpadded_grid_size (tuple[int, int]): unpad后特征网格的尺寸 (H_new, W_new)。
        newline_token_embedding (torch.Tensor): 换行符的特征向量 (D, )。

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
        - final_tokens: 包含图像和换行符token的最终序列 (1, K_final, D)。
        - final_indices: 包含图像和换行符全局ID的最终索引序列 (K_final, )。
    """
    # 假设Batch Size为1
    pruned_tokens = pruned_tokens.squeeze(0)
    pruned_indices = pruned_indices-576

    H_new, W_new = unpadded_grid_size
    num_newlines = H_new 

    if num_newlines <= 0:
        return pruned_tokens.unsqueeze(0), pruned_indices

    # 定义行边界索引 (第一行末尾的索引是 W_new-1, 第二行是 2*W_new-1, ...)
    row_boundaries = torch.arange(W_new, H_new*(W_new+1), W_new+1 ,device=pruned_indices.device)
    pruned_indices += (pruned_indices//W_new)
    # 使用searchsorted找到每个边界在已排序的pruned_indices中的插入点
    insertion_points = torch.searchsorted(pruned_indices, row_boundaries)

    final_tokens_list = []
    final_indices_list = []
    last_slice_start = 0

    for i in range(num_newlines):
        slice_end = insertion_points[i]
        
        # 添加当前行的幸存图像tokens和indices

        final_tokens_list.append(pruned_tokens[last_slice_start:slice_end])
        final_indices_list.append(pruned_indices[last_slice_start:slice_end])
        last_slice_start = slice_end

        # 添加换行符的token和全局ID
        final_tokens_list.append(newline_token_embedding.unsqueeze(0))
        final_indices_list.append(torch.tensor([row_boundaries[i]],device=pruned_indices.device))

    # # 添加最后一行的幸存tokens和indices
    # final_tokens_list.append(pruned_tokens[last_slice_start:])
    # final_indices_list.append(pruned_indices[last_slice_start:])
    
    final_tokens = torch.cat(final_tokens_list, dim=0).unsqueeze(0)
    final_indices = torch.cat(final_indices_list, dim=0)

    return final_tokens, final_indices+576

def encode_images_nuwa(self, images):
    image_features, keep_idx = self.get_model().get_vision_tower().forward(images)
    image_features = self.get_model().mm_projector(image_features)
    
    return image_features, keep_idx

def encode_images_nuwa_multi(self, images,image_sizes):
    image_features, keep_idx, unpad_size = self.get_model().get_vision_tower().forward(images,image_sizes)
    image_features = self.get_model().mm_projector(image_features)        
    return image_features, keep_idx,unpad_size
    
def restore_image_features_sorted(self, image_feature, cur_keep_idx, width, height):
   
    num_img, total_patches, feature_dim = image_feature.shape
    num_keep = cur_keep_idx.shape[1]  
    num_extra = total_patches - num_keep  


    cur_keep_idx_sorted, _ = cur_keep_idx.sort(dim=1)  # [num_img, num_keep]
    cur_keep_idx_sorted_restore = cur_keep_idx_sorted[:, 1:]-1

    restored_features = torch.zeros((num_img, 576, feature_dim), device=image_feature.device, dtype=image_feature.dtype)  # [num_img, total_patches, feature_dim]

    mask = torch.zeros(num_img, 576, dtype=torch.bool, device=image_feature.device)
    mask.scatter_(1, cur_keep_idx_sorted_restore, True)  

    kept_features = image_feature[:, 1:num_keep, :]  
    restored_features[mask] = kept_features.reshape(-1, feature_dim)  
    

    assert width * height == restored_features.shape[0], "width * height must equal num_img"
    restored_features = restored_features.view(height, width, 24, 24, feature_dim)  # [height, width, 24, 24, feature_dim]
    restored_features = restored_features.permute(0, 2, 1, 3, 4).contiguous()  # [height, 24, width, 24, feature_dim]
    restored_features = restored_features.view(height, 24, width * 24, feature_dim)  # [height, 24, width*24, feature_dim]
    restored_features = restored_features.view(height * 24, width * 24, feature_dim)  # [height*24, width*24, feature_dim]
    image_newline_expanded = self.model.image_newline.view(1, 1, feature_dim).expand(height * 24, 1, feature_dim).to(restored_features.device)  # [height*24, 1, feature_dim]
    grid_with_newline = restored_features

    mask = mask.view(height, width, 24, 24)  # [height, width, 24, 24]
    mask = mask.permute(0, 2, 1, 3).contiguous()  # [height, 24, width, 24]
    mask = mask.view(height * 24, width * 24)  # [height*24, width*24]

    mask_all = mask

    image_feature_select = grid_with_newline[mask_all]
    raw_img_feature_merge = image_feature[:,-num_extra:,].reshape(-1, feature_dim)
    cls_img_feature_merge = image_feature[:,0,]

    image_feature_select = torch.cat([image_feature_select, cls_img_feature_merge, raw_img_feature_merge])
    return image_feature_select

    
def prepare_inputs_labels_for_multimodal_nuwa(
    self, input_ids, position_ids, attention_mask, past_key_values, labels,
    images, modalities=["image"], image_sizes=None
):
    vision_tower = self.get_vision_tower()
    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        return input_ids, position_ids, attention_mask, past_key_values, None, labels
    if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            encoded_image_features, keep_idxs,unpad_size = self.encode_images_nuwa_multi(concat_images,image_sizes)
            
            self.position_offset = 576+ unpad_size[0]*unpad_size[1] + unpad_size[0]
            # image_features,all_faster_video_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)

            # This is a list, each element is [num_images, patch * patch, dim]
            # rank_print(f"Concat images : {concat_images.shape}")
            # pdb.set_trace()
            encoded_image_features = torch.split(encoded_image_features, [len(images_list)])
            
            
            image_features = []
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    image_features.append(self.get_2dPool(image_feat))
                else:
                    image_features.append(image_feat)
            # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            # rank_print(f"Encoded image feats : {[x.shape for x in image_features]}")
            # image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")
            
            mm_patch_merge_type = "new_line_flat"
            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type == "new_line_flat":
                new_image_features = []
                for image_feature in image_features:
                    global_indices = keep_idxs[keep_idxs<576]
                    global_feature = image_feature[:,:len(global_indices),:]
                    
                    local_indices = keep_idxs[len(global_indices):]
                    local_feature = image_feature[:,len(global_indices):,:]
                    local_feature,local_indices = insert_newlines_after_pruning(local_feature,local_indices,unpad_size,self.model.image_newline)
                    new_image_features.append(torch.cat([global_feature,local_feature],dim=1).squeeze(0))
                    
                    self.keep_idxs = torch.cat([global_indices,local_indices])
                image_features = new_image_features
            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    # import pdb; pdb.set_trace()
                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        if mm_newline_position == "grid":
                            # Grid-wise
                            image_feature = self.add_token_per_grid(image_feature)
                            if getattr(self.config, "add_faster_video", False):
                                faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                                # Add a token for each frame
                                concat_slow_fater_token = []
                                # import pdb; pdb.set_trace()
                                for _ in range(image_feature.shape[0]):
                                    if _ % self.config.faster_token_stride == 0:
                                        concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                    else:
                                        concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                # import pdb; pdb.set_trace()
                                image_feature = torch.cat(concat_slow_fater_token)

                                # print("!!!!!!!!!!!!")
                        
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            image_feature = self.add_token_per_frame(image_feature)

                            new_image_features.append(image_feature.flatten(0, 1))
                            
                        elif mm_newline_position == "one_token":
                            # one-token
                            image_feature = image_feature.flatten(0, 1)
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                            new_image_features.append(image_feature)      
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        # rank0_print("Single-images")
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                vision_tower_image_size=336
                                # raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                # rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        new_image_features.append(image_feature)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                        new_image_features.append(image_feature)
                image_features = new_image_features
                
            elif mm_patch_merge_type == "flat_spatial":
                new_image_features=[]
                new_keep_idxs=[]
                
                all_keep_idxs = torch.split(keep_idxs,[image_features[i].shape[0] for i in range(len(image_features))],dim=0)
                self.position_offset = keep_idxs.shape[0]*576
                
                for image_idx, image_feature in enumerate(image_features):
                    # 定义每行的token数量
                    keep_idxs = all_keep_idxs[image_idx]
                    
                    num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, 336)
                    TOKENS_PER_ROW = 24*num_patch_height
                    base_image_feature = image_feature[0]
                    base_keep_idxs = keep_idxs[0]
                    image_feature = image_feature[1:].view(-1,4096)  # torch.Size([4, 45, 4096])
                    keep_idxs = position_rebuild(keep_idxs[1:],num_patch_width, num_patch_height).flatten()
                    keep_idxs, sort_order = torch.sort(keep_idxs)
                    image_feature = image_feature[sort_order]
                    
                    new_features_list = []
                    new_idxs_list = []
                    
                    offset=0
                    if len(keep_idxs) > 0:
                        for i in range(len(keep_idxs) - 1):
                            # 添加当前的token特征和索引
                            try:
                                new_features_list.append(image_feature[i])
                                
                                current_row = keep_idxs[i] // TOKENS_PER_ROW
                                next_row = keep_idxs[i+1] // TOKENS_PER_ROW
                            except:
                                pdb.set_trace()
                            if next_row > current_row:
                                offset += 1
                                new_features_list.append(self.model.image_newline)
                                newline_idx = (current_row + 1) * TOKENS_PER_ROW
                                new_idxs_list.append(torch.tensor(newline_idx, device=keep_idxs.device, dtype=keep_idxs.dtype))
                                
                            new_idxs_list.append(keep_idxs[i]+offset)

                        new_features_list.append(image_feature[-1])
                        new_idxs_list.append(keep_idxs[-1]+offset)


                        image_feature = torch.cat([base_image_feature,torch.stack(new_features_list, dim=0)])
                        keep_idxs = torch.cat([base_keep_idxs,torch.stack(new_idxs_list, dim=0)+576])
                        
                        new_image_features.append(image_feature)
                        new_keep_idxs.append(keep_idxs)
                        
                        self.position_offset += offset
                        
                image_features = new_image_features
                self.keep_idxs = torch.hstack(new_keep_idxs)

            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
    else:
        image_features, keep_idxs = self.encode_images_nuwa(images)
        self.position_offset = 576
    self.keep_idxs = keep_idxs


    # TODO: image start / end is not implemented here to support pretraining.
    if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
        raise NotImplementedError

    # Let's just add dummy tensors if they do not exist,
    # it is a headache to deal with None all the time.
    # But it is not ideal, and if you have a better idea,
    # please open an issue / submit a PR, thanks.
    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    # remove the padding using attention_mask -- FIXME
    _input_ids = input_ids
    input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    new_input_embeds = []
    new_labels = []
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        if num_images == 0:
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue

        image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
        cur_new_input_embeds = []
        cur_new_labels = []

        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                cur_image_features = image_features[cur_image_idx]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

        cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

        new_input_embeds.append(cur_new_input_embeds)
        new_labels.append(cur_new_labels)

    # Truncate sequences to max length as image embeddings can make the sequence longer
    tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
    if tokenizer_model_max_length is not None:
        new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
        new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

    # Combine them
    max_len = max(x.shape[0] for x in new_input_embeds)
    batch_size = len(new_input_embeds)

    new_input_embeds_padded = []
    new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

    for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
            new_input_embeds_padded.append(torch.cat((
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                cur_new_embed
            ), dim=0))
            if cur_len > 0:
                new_labels_padded[i, -cur_len:] = cur_new_labels
                attention_mask[i, -cur_len:] = True
                position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
        else:
            new_input_embeds_padded.append(torch.cat((
                cur_new_embed,
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
            ), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded

    if _attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    if _position_ids is None:
        position_ids = None

    return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


 