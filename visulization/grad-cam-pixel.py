import torch
import torch.nn.functional as F
import numpy as np
import cv2
import cv2
import numpy as np
import torch
import sys
sys.path.append('./')
sys.path.append('../')
from torchvision.transforms import Compose, Normalize, ToTensor
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import argparse
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import GradCAM

from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from networks import get_model
from datasets import data_merge
from transformers import *
import os
import pandas as pd
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
def preprocess_image(
    img: np.ndarray, mean=[
        0.5, 0.5, 0.5], std=[
            0.5, 0.5, 0.5]) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)
# class GradCAM:
#     def __init__(self, model):
#         self.model = model
#         self.gradients = None
#         self.model.eval()

#     def save_gradient(self, module, grad_input, grad_output):
#         self.gradients = grad_output[0]

#     def forward(self, x):
#         return self.model(x)

#     def get_gradient(self):
#         return self.gradients

#     def __call__(self, x, target_layer):
#         # 注册钩子
#         h = target_layer.register_backward_hook(self.save_gradient)
#         # 前向传播
#         output = self.forward(x)
#         print('output', output)
#         # 清除梯度
#         self.model.zero_grad()
#         # 反向传播

#         output.backward()
#         # 移除钩子
#         h.remove()

#         return self.get_gradient()

def apply_colormap_on_image(image, activation, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.applyColorMap(np.uint8(255 * activation), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    cam = heatmap_resized + np.float32(image)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)



model = get_model()
use_adv = True
if use_adv:
    model.load_state_dict(torch.load('/root/autodl-tmp/results/DDG/results/2023-03-19-10:53:13/O_M_I_to_C/model/DDG_pO_M_I_to_C_best.pth')['state_dict'])
else:
    model.load_state_dict(torch.load('/root/autodl-tmp/results/DDG/results/2023-03-20-11:33:46/O_C_M_to_I/model/DDG_pO_M_I_to_C_best.pth')['state_dict'])
model = model.eval()
data_bank = data_merge('/root/autodl-tmp')
data_loader = torch.utils.data.DataLoader(data_bank, batch_size=16, shuffle=False, num_workers=16, pin_memory=True)

def get_iter(data_loader):
    sample = iter(data_loader).next()
    return sample['image_x'], sample['label'], sample['UUID'], sample['map_x']

# casia_dataset = data_bank.get_single_dataset('CASIA_MFSD', 'Train', transform=None)
# casia_loader = torch.utils.data.DataLoader(casia_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

# rgb_img, label, _, _ = get_iter(casia_loader)

out_path = './grad-cam-images'
if not os.path.exists(out_path):
    os.makedirs(out_path)

img_name = 'spoof-origin-3'
img_path = os.path.join(out_path, img_name + '.png')
rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
print('rgb_img', rgb_img.shape)
cv2.imwrite(os.path.join(out_path, 'origin.png'), np.array(rgb_img))

rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

# 选择目标层
target_layer = model.layer4  # 或其他层，例如 model.features[-1]

# 初始化Grad-CAM
grad_cam = GradCAM(model, target_layer)
targets = [ClassifierOutputTarget(0)]

# 应用彩色映射
for i in range(input_tensor.shape[0]):
    # 计算梯度
    grads = grad_cam(input_tensor)
    grads = grads[0, :]
    print('grads', grads.shape)
    # 计算Grad-CAM
    # cam = F.relu(torch.mean(grads, dim=1)).data.cpu().numpy()

    # 归一化
    # cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    # vis_image = apply_colormap_on_image(input_tensor[i].numpy().transpose(1, 2, 0), grads[i])

    vis_image = show_cam_on_image(rgb_img, grads, use_rgb=True)
    print('vis_image', vis_image.shape)
    # vis_images.append(vis_image)

    # cv2.imwrite(os.path.join(out_path, str(i) + '_' + 'origin.png'), np.array(zero_samples_1[0]))
    cv2.imwrite(os.path.join(out_path, img_name + '_' + 'grad-cam.png'), vis_image)
