#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
# export 多张 GPU ！！！ export CUDA_VISIBLE_DEVICES=4,5,6,7

#emotion_list=("awe" "contentment" "excitement" "anger" "disgust" "fear" "sadness")
emotion_list=("amusement")

for emotion in "${emotion_list[@]}"
do
echo "$emotion" # 在Bash脚本中，echo "$emotion" 这行代码的意思是打印变量 emotion 的值到标准输出（通常是终端或控制台）。

accelerate launch \
--config_file /mnt/d/run_config.yaml /mnt/d/models/conceptor/one_step_reconstruction.py \
--train_data_dir=./emotion_train/ \
--output_dir=/mnt/d/models/conceptor/emotion \
--path_to_encoder_embeddings="./clip_text_encoding.pt" \
--name="cake" \
--placeholder_token="<>" \
--max_train_steps=30 \
--num_train_epochs=30 \
--prompt="a photo of cake" \
--repeats=1 \
--train_batch_size=10 \
--dictionary_size=1000 \
--gradient_accumulation_steps=1 \
--learning_rate=5e-4 \
--pretrained_model_name_or_path=/mnt/d/models/stable-diffusion-2-1 \
--seed=114514 \
--validation_prompt="a photo of a <>"

accelerate launch \
--config_file /mnt/d/run_config.yaml /mnt/d/models/conceptor/Visualize_Concept.py \
--name="cake" \

done

#--remove_concept_tokens \ 出现就是 True
# remove_concept_tokens 第一步筛选词表时，是否去除相似概念
# bug!!! 少了 \ 后面参数全部传不进去！！！

#python /mnt/d/bigProject/utils/generate.py \
# --ti_dir=/mnt/d/bigProject/results/train_ti/20240711/train/amusement \
# --dest_dir=/mnt/d/bigProject/results/train_ti/20240711/test/amusement \
# --prompt="a photo that evokes <amusement>" \
# --image_num=4 \
# --min_step=0 \
# --max_step=100000 \
# --step_size=50