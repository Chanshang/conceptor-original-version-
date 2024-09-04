#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
# export 多张 GPU ！！！ export CUDA_VISIBLE_DEVICES=4,5,6,7

#emotion_list=("awe" "contentment" "excitement" "anger" "disgust" "fear" "sadness")
emotion_list=("amusement")

for emotion in "${emotion_list[@]}"
do
echo "$emotion" # 在Bash脚本中，echo "$emotion" 这行代码的意思是打印变量 emotion 的值到标准输出（通常是终端或控制台）。
accelerate launch \
--config_file /mnt/d/models/run_config.yaml /mnt/d/models/conceptor/one_step_reconstruction.py \
--train_data_dir=/mnt/d/dataset/train_by_emotion/"$emotion" \
--output_dir=/mnt/d/models/conceptor/emotion \
--name="nmsl!!!"
--placeholder_token="<>" \
--max_train_steps=10 \
--repeats=1 \
--learnable_property=style \
--train_batch_size=10 \
--gradient_accumulation_steps=4 \
--learning_rate=5e-4 \
--pretrained_model_name_or_path=/mnt/d/models/stable-diffusion-2-1 \
--seed=114514 \
--validation_prompt="a photo of a <>"
done

echo "Train batch size: $train_batch_size"
echo "Gradient accumulation steps: $gradient_accumulation_steps"
echo "Learning rate: $learning_rate"

#python /mnt/d/bigProject/utils/generate.py \
# --ti_dir=/mnt/d/bigProject/results/train_ti/20240711/train/amusement \
# --dest_dir=/mnt/d/bigProject/results/train_ti/20240711/test/amusement \
# --prompt="a photo that evokes <amusement>" \
# --image_num=4 \
# --min_step=0 \
# --max_step=100000 \
# --step_size=50