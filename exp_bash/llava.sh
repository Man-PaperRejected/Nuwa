#!/bin/bash

source activate nuwa

export CUDA_VISIBLE_DEVICES=0

# 定义要迭代的 agg_layer 值范围
# for layer in {2..16}
# "gqa" "mmmu_val" "pope"    "textvqa_val" "scienceqa_img" "mmvet" "vqav2_val" "mmbench_en_dev"  "mme" 
for nt in 64
do
  for task in "refcoco_bbox_rec_test"  "refcoco+_bbox_rec_testA" "refcoco+_bbox_rec_testB" "refcocog_bbox_rec_test" "mme"  "mmbench_en_dev"  
    do
      echo "Running for ${task} Distance 280 nt=${nt}"

      # 构建 output_path
      export DIST=280
      export NT=${nt}
      output_dir="./logs/main_experiment/token64"

      python3 -m accelerate.commands.launch  \
        --num_processes=1      -m lmms_eval     --model llava   \
        --model_args pretrained="/data/model/llava-v1.5-7b"   \
        --tasks ${task}     --batch_size 1     --log_samples     --log_samples_suffix llava_v1.5_${task}   --output_path "${output_dir}"

      echo "----------------------------------------"
  done
done

# "vqav2_val" "refcoco_bbox_rec_val"  "refcoco+_bbox_rec_val"  "refcocog_bbox_rec_val"
for nt in 128
do
  for task in "gqa" "mmmu_val" "pope"    "textvqa_val" "scienceqa_img" "mmvet"   "mmbench_en_dev"  "mme" "refcoco_bbox_rec_test"  "refcoco+_bbox_rec_testA" "refcoco+_bbox_rec_testB" "refcocog_bbox_rec_test"
    do
      echo "Running for ${task} Distance 280 nt=${nt}"

      # 构建 output_path
      export DIST=280
      export NT=${nt}
      output_dir="./logs/main_experiment/token128"

      python3 -m accelerate.commands.launch  \
        --num_processes=4      -m lmms_eval     --model llava   \
        --model_args pretrained="/data/model/llava-v1.5-7b"   \
        --tasks ${task}     --batch_size 1     --log_samples     --log_samples_suffix llava_v1.5_${task}   --output_path "${output_dir}"

      echo "----------------------------------------"
  done
done

for nt in 192
do
  for task in "gqa" "mmmu_val" "pope"    "textvqa_val" "scienceqa_img" "mmvet"   "mmbench_en_dev"  "mme" "refcoco_bbox_rec_test"  "refcoco+_bbox_rec_testA" "refcoco+_bbox_rec_testB" "refcocog_bbox_rec_test"
    do
      echo "Running for ${task} Distance 280 nt=${nt}"

      # 构建 output_path
      export DIST=280
      export NT=${nt}
      output_dir="./logs/main_experiment/token192"

      python3 -m accelerate.commands.launch  \
        --num_processes=4      -m lmms_eval     --model llava   \
        --model_args pretrained="/data/model/llava-v1.5-7b"   \
        --tasks ${task}     --batch_size 1     --log_samples     --log_samples_suffix llava_v1.5_${task}   --output_path "${output_dir}"

      echo "----------------------------------------"
  done
done
# # "gqa" "mmbench_en_dev"  "mme"  "mmmu_val" "pope"    "textvqa_val" "scienceqa_img" llll"mmvet"  
# for cfg in 18 148 280 412 544 676 808 940 1058
# do 
#   for task in  "gqa" "mmbench_en_dev"  "mme" "refcoco_bbox_rec_test"  "refcoco+_bbox_rec_testA" "refcoco+_bbox_rec_testB" "refcocog_bbox_rec_test" 
#   do
#     echo "Running for ${task} Distance ${cfg}"

#     # 构建 output_path
#     export DIST=${cfg}
#     output_dir="./logs/ablition/dist/dist${cfg}"

#     python3 -m accelerate.commands.launch  \
#       --num_processes=5      -m lmms_eval     --model llava   \
#       --model_args pretrained="/data/model/llava-v1.5-7b"   \
#       --tasks ${task}     --batch_size 1     --log_samples     --log_samples_suffix llava_v1.5_${task}   --output_path "${output_dir}"  

#     echo "----------------------------------------"

#   done
# done
echo "All iterations completed."
python3 -c "import torch; torch.cuda.empty_cache()"


# python3 -m accelerate.commands.launch  \
#     --num_processes=2     -m lmms_eval     --model llava   \
#     --model_args pretrained="/data/model/llava-v1.5-7b"   \
#     --tasks gqa     --batch_size 1     --log_samples     --log_samples_suffix llava_v1.5_gqa   --output_path ./log