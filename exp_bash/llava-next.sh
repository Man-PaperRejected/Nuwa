#!/bin/bash

source activate nuwa

export CUDA_VISIBLE_DEVICES=7

# 定义要迭代的 agg_layer 值范围
# for layer in {2..16}
   #     --model_args pretrained="/data/model/llava-v1.5-7b"   \ pretrained="/data/model/llava-v1.6-vicuna-7b,conv_template=llava_v1"
# "refcoco_bbox_rec_test"  "refcoco+_bbox_rec_testA" "refcoco+_bbox_rec_testB" "refcocog_bbox_rec_test"
for nt in 160 320 640
do
  for task in  "gqa" "pope"  "textvqa_val" "scienceqa_img"   "mmbench_en_dev"  "mme"
  do
    for dist in 280
    do
      for thre in  0.5
      do
        echo "Running for ${task} Distance ${dist} THRE ${thre}  nt=${nt} "

        # 构建 output_path
        export DIST=${dist}
        export NT=${nt}
        export THRE=${thre}
        output_dir="./logs/token${nt}/${DIST}_${THRE}"

        python3 -m accelerate.commands.launch  \
          --num_processes=4      -m lmms_eval     --model llava   \
          --model_args pretrained="/data/model/llava-v1.6-vicuna-7b,conv_template=llava_v1"   \
          --tasks ${task}     --batch_size 1     --log_samples     --log_samples_suffix llava_v1.5_${task}   --output_path "${output_dir}"  

        echo "----------------------------------------"
        done
      done
  done
done



# for nt in 160
# do
#   for task in "mmmu_val" "pope"  "textvqa_val"  "mmbench_en_dev"  "mme" "refcoco_bbox_rec_test" "refcoco_bbox_rec_test"  "refcoco+_bbox_rec_testA"  "refcocog_bbox_rec_test"
#   do
#       echo "Running for ${task} Distance 140 nt=${nt}"

#       # 构建 output_path
#       export DIST=280
#       export NT=${nt}
#       output_dir="./logs/llava-next/token160"

#       python3 -m accelerate.commands.launch  \
#         --num_processes=5      -m lmms_eval     --model llava   \
#         --model_args pretrained="/data/model/llava-v1.6-vicuna-7b,conv_template=llava_v1"   \
#         --tasks ${task}     --batch_size 1     --log_samples     --log_samples_suffix llava_v1.5_${task}   --output_path "${output_dir}"  --limit 600

#       echo "----------------------------------------"
#   done
# done