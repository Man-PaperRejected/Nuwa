import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from typing import Any, Optional, Tuple, Union, List
from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPAttention, CLIPVisionTransformer, CLIPEncoder
from transformers.models.clip.modeling_clip import eager_attention_forward,logger
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput

import pdb
def attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    causal_attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Input shape: Batch x Time x Channel"""
    
    output_cls_attn_weights = getattr(self,"output_cls_attn_weights", None)
    batch_size, seq_length, embed_dim = hidden_states.shape

    queries = self.q_proj(hidden_states)
    keys = self.k_proj(hidden_states)
    values = self.v_proj(hidden_states)

    queries = self._shape(queries, seq_length, batch_size)
    keys = self._shape(keys, seq_length, batch_size)
    values = self._shape(values, seq_length, batch_size)
    
    if self.config._attn_implementation == "flash_attention_2":
        self.is_causal = causal_attention_mask is not None
    else:
        if attention_mask is not None and causal_attention_mask is not None:
            attention_mask = attention_mask + causal_attention_mask
        elif causal_attention_mask is not None:
            attention_mask = causal_attention_mask


    attn_weights = None
    if output_cls_attn_weights:

        cls_token_query = queries[:, :, 0:1, :] * self.scale
        

        cls_attn_scores = torch.matmul(cls_token_query, keys.transpose(-1, -2))
        
        if attention_mask is not None:
            cls_attn_scores = cls_attn_scores + attention_mask
        
        cls_attn_weights = nn.functional.softmax(cls_attn_scores, dim=-1)
        
        attn_weights = cls_attn_weights

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":

        if self.config._attn_implementation == "sdpa" and output_attentions and not output_cls_attn_weights:
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                '"eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS.get(self.config._attn_implementation, eager_attention_forward)

    attn_output, _ = attention_interface(
        self,
        queries,
        keys,
        values,
        attention_mask,
        is_causal=self.is_causal,
        scaling=self.scale,
        dropout=0.0 if not self.training else self.dropout,
        output_attentions=False,
    )

    attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, embed_dim)
    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights

def CLIPAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    causal_attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""

    bsz, tgt_len, embed_dim = hidden_states.size()

    # get query proj
    query_states = self.q_proj(hidden_states) * self.scale
    key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    key_states = key_states.view(*proj_shape)
    value_states = value_states.view(*proj_shape)

    src_len = key_states.size(1)
    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

    if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
            f" {attn_weights.size()}"
        )

    # apply the causal_attention_mask first
    if causal_attention_mask is not None:
        if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                f" {causal_attention_mask.size()}"
            )
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)



    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if output_attentions:
        # this operation is a bit akward, but it's required to
        # make sure that attn_weights keeps its gradient.
        # In order to do so, attn_weights have to reshaped
        # twice and have to be reused in the following
        attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
    else:
        attn_weights_reshaped = None

    attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

    attn_output = torch.bmm(attn_probs, value_states)

    if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights_reshaped

def CLIP_EncoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    causal_attention_mask: torch.Tensor,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.FloatTensor]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`): attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            `(config.encoder_attention_heads,)`.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
    """
    residual = hidden_states

    hidden_states = self.layer_norm1(hidden_states)


    hidden_states, attn_weights = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
    )
    
    hidden_states = residual + hidden_states
    
    # r = self._info["r"].pop(0)
    # if r > 0:
    #     self.metric = metric    
    residual = hidden_states
    hidden_states = self.layer_norm2(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (attn_weights,)

    return outputs


def parse_r(num_layers: int, r: Union[List[int], Tuple[int, float], int]) -> List[int]:
    """
    Copy from the TOME. 
    https://github.com/facebookresearch/ToMe

    Process a constant r or r schedule into a list for use internally.

    r can take the following forms:
     - int: A constant number of tokens per layer.
     - Tuple[int, float]: A pair of r, inflection.
       Inflection describes there the the reduction / layer should trend
       upward (+1), downward (-1), or stay constant (0). A value of (r, 0)
       is as providing a constant r. (r, -1) is what we describe in the paper
       as "decreasing schedule". Any value between -1 and +1 is accepted.
     - List[int]: A specific number of tokens per layer. For extreme granularity.
    """
    inflect = 0
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)
    elif isinstance(r, tuple):
        r, inflect = r

    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)

    return [int(min_val + step * i) for i in range(num_layers)]

def make_tome_class(transformer_class):
    class NuwaTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """
            
        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._info["r"] = parse_r(len(self.vision_model.encoder.layers), self.r)
            # self._info["r"] = self.r

            self._info["size"] = None
            self._info["source"] = None

            return super().forward(*args, **kwdargs)

    return NuwaTransformer

def apply_info(model, matain_token, distance):

    Stage2_CONFIGS={
            64:16,
            128:32,
            192:48,
        }
    
    vision_tower = model.model.vision_tower.vision_tower
    # return cls-attn from the second last layer
    vision_tower.vision_model.encoder.layers[-2].self_attn.output_cls_attn_weights = True
    vision_tower._info = {
        "matain_token":matain_token,
        "distance":distance,
    }
    for module in model.modules():
        if isinstance(module, CLIPEncoderLayer):
            module._info = vision_tower._info

