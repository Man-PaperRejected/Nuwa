from .utils import CLIP_EncoderLayer_forward, CLIPAttention_forward, apply_info,attn_forward
from .clip_encoder import CLIPVisionTower_Nuwa,CLIPVisionTower_Nuwa_Next,CLIPVisionTower_Nuwa_abli
from .llava_arch import prepare_inputs_labels_for_multimodal_nuwa, encode_images_nuwa, encode_images_nuwa_multi, restore_image_features_sorted

def nuwa(model, matain_token=64, distance=280):

    apply_info(model, matain_token=matain_token, distance=distance)


    from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPAttention

    CLIPEncoderLayer.forward = CLIP_EncoderLayer_forward
    CLIPAttention.forward = CLIPAttention_forward

    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
    # CLIPVisionTower.forward = CLIPVisionTower_Nuwa_abli.forward
    CLIPVisionTower.forward = CLIPVisionTower_Nuwa.forward

    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower

    from llava.model.llava_arch import LlavaMetaForCausalLM
    if hasattr(LlavaMetaForCausalLM, 'prepare_inputs_labels_for_multimodal'):
        LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal_nuwa
        LlavaMetaForCausalLM.restore_image_features_sorted = restore_image_features_sorted
        LlavaMetaForCausalLM.encode_images_nuwa_multi = encode_images_nuwa_multi
        LlavaMetaForCausalLM.encode_images_nuwa = encode_images_nuwa


    return model
