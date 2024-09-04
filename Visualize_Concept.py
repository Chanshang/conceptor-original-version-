from diffusers import StableDiffusionPipeline
from diffusers.schedulers import LMSDiscreteScheduler
from random import randrange
import torch
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import (
    LMSDiscreteScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    PNDMScheduler,
)
import torchvision.transforms as transforms
from random import randrange
import requests
import torch.optim as optim
from PIL import Image, ImageDraw
import numpy as np
import argparse
import os
import glob
from pathlib import Path

import argparse
# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='Process some parameters.')
# 添加参数
parser.add_argument('--name', type=str, help="president", default="christmas-tree-test")
# 解析命令行参数
args = parser.parse_args()
# 使用传递的参数
print(f"The value of name is: {args.name}")


# 词表只保存一次
# initialize stable diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained("../stable-diffusion-2-1")
pipe.to("cuda")
scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = scheduler
orig_embeddings = (
    pipe.text_encoder.text_model.embeddings.token_embedding.weight.clone().detach()
)
pipe.text_encoder.text_model.embeddings.token_embedding.weight.requires_grad_(False)


concept = "amusement"
folder = "./emotion"

file_path = f"{folder}/{args.name}.pt"

# load coefficients  参数
alphas_dict = torch.load(file_path).detach_().requires_grad_(False)
# load vocabulary  词表
dictionary = torch.load("./dictionary.pt")

# 排序
sorted_alphas, sorted_indices = torch.sort(alphas_dict, descending=True)

num_indices = 50
top_indices_orig_dict = [dictionary[i] for i in sorted_indices[:num_indices]]
print("top coefficients: ", sorted_alphas[:num_indices].cpu().numpy())
alpha_ids = [pipe.tokenizer.decode(idx) for idx in top_indices_orig_dict]
print("top tokens: ", alpha_ids)

# 打开文件，准备写入, open 会自动创建 文件，若已存在则清空
top_alphas = sorted_alphas[:num_indices].cpu().numpy()
txt_position = f"{folder}/{args.name}.txt"

with open(txt_position, 'w') as file:
    # 写入 top coefficients
    file.write("top coefficients: " + str(top_alphas) + '\n')
    # 写入 top tokens
    file.write("top tokens:  ")
    for i, token in enumerate(alpha_ids):
        if (i + 1) % 10 == 0:
            file.write(f"{token}\n")  # 在每十个元素后换行
        else:
            file.write(f"{token}  ")  # 在同一行打印元素，以空格分隔

num_tokens = 50
alphas = torch.zeros(orig_embeddings.shape[0]).cuda()
sorted_alphas, sorted_indices = torch.sort(alphas_dict.abs(), descending=True)
top_word_idx = [dictionary[i] for i in sorted_indices[:num_tokens]]
for i, index in enumerate(top_word_idx):
    alphas[index] = alphas_dict[sorted_indices[i]]


# add placeholder for w^*
placeholder_token = "<>"
pipe.tokenizer.add_tokens(placeholder_token)
placeholder_token_id = pipe.tokenizer.convert_tokens_to_ids(placeholder_token)
pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
token_embeds = (
    pipe.text_encoder.get_input_embeddings().weight.detach().requires_grad_(False)
)

# compute w^* and normalize its embedding
learned_embedding = torch.matmul(alphas, orig_embeddings).flatten()
norms = [i.norm().item() for i in orig_embeddings]
avg_norm = np.mean(norms)
learned_embedding /= learned_embedding.norm()
learned_embedding *= avg_norm

# add w^* to vocabulary
token_embeds[placeholder_token_id] = torch.nn.Parameter(learned_embedding)

import math

def get_image_grid(images) -> Image:
    num_images = len(images)
    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))
    width, height = images[0].size
    grid_image = Image.new("RGB", (cols * width, rows * height))
    for i, img in enumerate(images):
        x = i % cols
        y = i // cols
        grid_image.paste(img, (x * width, y * height))
    return grid_image

generator = torch.Generator("cuda").manual_seed(0)

prompt = "a photo of a <>"
image = pipe(
    prompt,
    guidance_scale=7.5,
    generator=generator,
    return_dict=False,
    num_images_per_prompt=6,
    num_inference_steps=50,
)
im = get_image_grid(image[0])
output_path = f'./Save_Images/{args.name}.jpg'
# 保存图片
im.save(output_path)


# prompt = "a photo of an excitement"
# generator = torch.Generator("cuda").manual_seed(0)
# image = pipe(
#     prompt,
#     guidance_scale=7.5,
#     generator=generator,
#     return_dict=False,
#     num_images_per_prompt=6,
#     num_inference_steps=50,
# )
# im = get_image_grid(image[0])
# output_path = f'./Save_Images/{prompt}.jpg'
# # 保存图片
# im.save(output_path)
