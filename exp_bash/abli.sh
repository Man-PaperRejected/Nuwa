source activate nuwa

export CUDA_VISIBLE_DEVICES=0,1,5,6,7

# for task in "refcocog_bbox_rec_test"
# do
# # 构建 output_path
# export REGION="True"
# export L2NORM="False"
# export RANDOM="True"
# export DIST=280
# export NT=128
# output_dir="./logs/ablition/rg_${region}_l2_${l2norm}_random_${random}"
# echo "Running for ${task} Distance 280 nt=${nt}  REGION=${region}  L2NORM=${l2norm} random=${random}"

# python3 -m accelerate.commands.launch  \
#     --num_processes=5      -m lmms_eval     --model llava   \
#     --model_args pretrained="/data/model/llava-v1.5-7b"   \
#     --tasks ${task}     --batch_size 1     --log_samples     --log_samples_suffix llava_v1.5_${task}   --output_path "${output_dir}"

# echo "----------------------------------------"
# done

for task in "refcoco+_bbox_rec_testB" "refcocog_bbox_rec_test"
do
export REGION="True"
export L2NORM="True"
export RANDOM="False"
export DIST=280
export NT=128
output_dir="./logs/ablition/test"
echo "Running for ${task} Distance 280 nt=${nt}  REGION=${region}  L2NORM=${l2norm} random=${random}"

python3 -m accelerate.commands.launch  \
    --num_processes=5      -m lmms_eval     --model llava   \
    --model_args pretrained="/data/model/llava-v1.5-7b"   \
    --tasks ${task}     --batch_size 1     --log_samples     --log_samples_suffix llava_v1.5_${task}   --output_path "${output_dir}"

echo "----------------------------------------"
done


for nt in 128
do
    for region in "True" "False"
    do
        for l2norm in "True" "False"
            do
                for random in  "False" "True"
                do
                for task in 'gqa' "refcoco+_bbox_rec_testA"  "mmbench_en_dev"  "mme" "refcoco_bbox_rec_test"  "refcoco+_bbox_rec_testB" "refcocog_bbox_rec_test"
                    do
                    

                    # 构建 output_path
                    export REGION=${region}
                    export L2NORM=${l2norm}
                    export RANDOM=${random}
                    export DIST=280
                    export NT=${nt}
                    output_dir="./logs/ablition/rg_${region}_l2_${l2norm}_random_${random}"
                    echo "Running for ${task} Distance 280 nt=${nt}  REGION=${region}  L2NORM=${l2norm} random=${random}"

                    python3 -m accelerate.commands.launch  \
                        --num_processes=4      -m lmms_eval     --model llava   \
                        --model_args pretrained="/data/model/llava-v1.5-7b"   \
                        --tasks ${task}     --batch_size 1     --log_samples     --log_samples_suffix llava_v1.5_${task}   --output_path "${output_dir}"

                    echo "----------------------------------------"
                done
            done
        done
    done
done
