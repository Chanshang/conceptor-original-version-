# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A script for saving all the word embeddings."""
# pylint: disable=g-multiple-import,g-importing-member,g-bad-import-order,missing-function-docstring

import argparse

from diffusers import StableDiffusionPipeline
from diffusers.schedulers import LMSDiscreteScheduler
import torch
from transformers import CLIPModel, CLIPProcessor

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="../stable-diffusion-2-1/",
        help=(
            "Path to pretrained model or model identifier from"
            " huggingface.co/models."
        ),
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="../clip-vit-base-patch32",  # patch 切的小比较厉害，操作图片更大
        help=(
            "The CLIP model to use for the calculation of the image-text" " matching."
        ),
    )
    parser.add_argument(
        "--path_to_encoder_embeddings",
        type=str,
        default="./clip_text_encoding.pt",
        help="Path to the saved embeddings matrix of the text encoder",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # model 和 processor 都是 CLIP 的
    model = CLIPModel.from_pretrained(args.clip_model).cuda()
    processor = CLIPProcessor.from_pretrained(args.clip_model)

    # initialize stable diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path)
    pipe.to("cuda")
    scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = scheduler

    orig_embeddings = (
        pipe.text_encoder.text_model.embeddings.token_embedding.weight.clone().detach()
    )

    print("\n orig_embeddings.shape!!!\n")
    print(orig_embeddings.shape)

    # orig_embeddings 是从稳定扩散（Stable Diffusion）模型的文本编码器部分提取的原始嵌入向量。
    # orig_embeddings 是一个包含了模型中所有 token 嵌入向量的矩阵。每一行对应于模型词汇表中的一个 token 的嵌入向量。
    # .clone(): 这个方法创建权重矩阵的一个副本。
    # .detach(): 这个方法将权重矩阵从计算图中分离出来，意味着后续对这个矩阵的操作不会计算梯度，也不会影响模型的训练过程。

    imagenet_templates = [
        "a photo of a {}",
        # "a rendering of a {}",
        # "a cropped photo of the {}",
        # "the photo of a {}",
        # "a photo of a clean {}",
        # "a photo of a dirty {}",
        # "a dark photo of the {}",
        # "a photo of my {}",
        # "a photo of the cool {}",
        # "a close-up photo of a {}",
        # "a bright photo of the {}",
        # "a cropped photo of a {}",
        # "a photo of the {}",
        # "a good photo of the {}",
        # "a photo of one {}",
        # "a close-up photo of the {}",
        # "a rendition of the {}",
        # "a photo of the clean {}",
        # "a rendition of a {}",
        # "a photo of a nice {}",
        # "a good photo of a {}",
        # "a photo of the nice {}",
        # "a photo of the small {}",
        # "a photo of the weird {}",
        # "a photo of the large {}",
        # "a photo of a cool {}",
        # "a photo of a small {}",
    ]

    def get_embedding_for_prompt(prompt, templates):
        with torch.no_grad():
            texts = [
                template.format(prompt) for template in templates  # 植入prompt
            ]  # format with class
            text_preprocessed = processor(text=texts, return_tensors="pt", padding=True) # 用 CLIP 加工
            text_encodings = model.get_text_features(
                input_ids=text_preprocessed["input_ids"].cuda(),
                attention_mask=text_preprocessed["attention_mask"].cuda(),
            )
            text_encodings /= text_encodings.norm(dim=-1, keepdim=True)
            text_encodings = text_encodings.mean(dim=0)  # 取了平均
            text_encodings /= text_encodings.norm()
            return text_encodings.float()

        # 这个函数返回的是一个单一的嵌入向量，它的维度与单个文本嵌入向量的维度相同，而不会因为文本数量的增加而变大。这个向量代表了输入 prompt 在所有模板下的综合特征。

    top_encodings_open_clip = [
        # token：整数
        # decoder[token]：文本字符串
        # pipe.tokenizer.decoder[token]：文本嵌入向量
        get_embedding_for_prompt(pipe.tokenizer.decoder[token], imagenet_templates) for token in range(orig_embeddings.shape[0])
    ]

    top_encodings_open_clip = torch.stack(top_encodings_open_clip, dim=0)

    # print(top_encodings_open_clip.shape)
    # torch.Size([49408, 512])  !!!

    torch.save(top_encodings_open_clip, args.path_to_encoder_embeddings)

if __name__ == "__main__":
    main()
