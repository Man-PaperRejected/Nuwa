#!/bin/bash

source activate nuwa
export CUDA_VISIBLE_DEVICES=7



# accelerate launch \
#     --num_processes=4 --main_process_port=12346 -m lmms_eval \
#     --model qwen2_5_vl \
#     --model_args=pretrained=/data/model/Qwen2.5-VL-7B-Instruct,max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=False \
#     --tasks refcoco_bbox_rec_val \
#     --batch_size 1  --log_samples     --log_samples_suffix llava_v1.5_${task}   --output_path "./log/nuwa"  --limit 100


for ratio in 0.111 0.222 0.333
do
  for task in "gqa"  "pope"    "textvqa_val" "scienceqa_img" "mmvet" "vqav2_val" "refcoco_bbox_rec_val"  "refcoco+_bbox_rec_val"  "refcocog_bbox_rec_val" "mmbench_en_dev"  "mme" "refcoco_bbox_rec_test"  "refcoco+_bbox_rec_testA" "refcoco+_bbox_rec_testB" "refcocog_bbox_rec_test"
    do
      echo "Running for ${task}  ratio=${ratio}"

      # 构建 output_path
      export KEEP_RATIO=${ratio}
      output_dir="./logs/qwen/token_${ratio}"

      python3 -m accelerate.commands.launch  \
            --num_processes=1 --main_process_port=12346 -m lmms_eval \
            --model qwen2_5_vl \
            --model_args=pretrained=/data/model/Qwen2.5-VL-7B-Instruct,max_pixels=12845056,interleave_visuals=False \
            --tasks ${task} \
            --batch_size 1  --log_samples     --log_samples_suffix llava_v1.5_${task}   --output_path "./log/nuwa"   --limit 10

      echo "----------------------------------------"
  done
done



# 定义要迭代的 agg_layer 值范围
# for layer in {2..16}
#    "refcoco_bbox_rec_test" "gqa""gqa" "mmbench_en_dev" "mme"  "refcoco_bbox_rec_test""mmvet" "gqa" "mmbench_en_dev" "mme"  "refcoco_bbox_rec_test"    "textvqa_val" "scienceqa_img" "mmmu_val" "pope"
# "refcoco_bbox_rec_test"  "refcoco+_bbox_rec_testA" "refcoco+_bbox_rec_testB" "refcoco+_bbox_rec_val" "refcocog_bbox_rec_test" 
# "refcoco_bbox_rec_test"



# for task in "refcoco_bbox_rec_test"  "refcoco+_bbox_rec_testA" "refcoco+_bbox_rec_testB" "refcoco+_bbox_rec_val" "refcocog_bbox_rec_test"  "refcocog_bbox_rec_test"  "mmvet"   "vqav2_val" "gqa" "mmbench_en_dev" "mme"  "mmmu_val" "pope"     "textvqa_val" "scienceqa_img" 
# do
#   echo "Running for ${task}"

#   # 构建 output_path
#   output_dir="./logs/token64_flat/visionzip"

#   python3 -m accelerate.commands.launch  \
#     --num_processes=1      -m lmms_eval     --model llava   \
#     --model_args pretrained="/data/model/llava-v1.5-7b"   \
#     --tasks ${task}     --batch_size 1     --log_samples     --log_samples_suffix llava_v1.5_${task}   --output_path "${output_dir}"  

#   echo "----------------------------------------"

# done

# echo "All iterations completed."
# python3 -c "import torch; torch.cuda.empty_cache()"


# python3 -m accelerate.commands.launch  \
#     --num_processes=2     -m lmms_eval     --model llava   \
#     --model_args pretrained="/data/model/llava-v1.5-7b"   \
#     --tasks gqa     --batch_size 1     --log_samples     --log_samples_suffix llava_v1.5_gqa   --output_path ./log