import cv2
import numpy as np
import torch
import sys
sys.path.append('../')
from networks import get_model
from datasets import data_merge
from transformers import *
import os
import pandas as pd

model = get_model()
use_adv = True
if use_adv:
    model.load_state_dict(torch.load('/root/autodl-tmp/results/DDG/results/2023-03-19-10:53:13/O_M_I_to_C/model/DDG_pO_M_I_to_C_best.pth')['state_dict'])
else:
    model.load_state_dict(torch.load('/root/autodl-tmp/results/DDG/results/2023-03-20-11:33:46/O_C_M_to_I/model/DDG_pO_C_M_to_I_best.pth')['state_dict'])
model = model.eval()
data_bank = data_merge('/root/autodl-tmp')
data_loader = torch.utils.data.DataLoader(data_bank, batch_size=16, shuffle=False, num_workers=16, pin_memory=True)

def get_iter(data_loader):
    sample = iter(data_loader).next()
    return sample['image_x'], sample['label'], sample['UUID'], sample['map_x']

def extract_data_labels(data_loader):
    data_list = []
    label_list = []
    for sample_batched in data_loader:
        data, label, uuid, map_x = sample_batched['image_x'], sample_batched['label'], sample_batched['UUID'], sample_batched['map_x']
        data_list.append(data)
        label_list.append(label)
    data = torch.cat(data_list, dim=0)
    label = torch.cat(label_list, dim=0)
    zero_indices = (label.squeeze(1) == 0).nonzero().squeeze()
    one_indices = (label.squeeze(1) == 1).nonzero().squeeze()
    zero_data = data[zero_indices]
    one_data = data[one_indices]
    zero_sample_indices = torch.randperm(zero_data.shape[0])[:20]
    one_sample_indices = torch.randperm(one_data.shape[0])[:20]
    zero_samples = zero_data[zero_sample_indices]
    one_samples = one_data[one_sample_indices]
    return zero_samples, one_samples


casia_dataset = data_bank.get_single_dataset('CASIA_MFSD', 'Train', transform=transformer_train_pure())
casia_loader = torch.utils.data.DataLoader(casia_dataset, batch_size=32, shuffle=False, num_workers=1, pin_memory=True)

oulu_dataset = data_bank.get_single_dataset('OULU', 'Train', transform=transformer_train_pure())
oulu_loader = torch.utils.data.DataLoader(oulu_dataset, batch_size=32, shuffle=False, num_workers=1, pin_memory=True)

msu_dataset = data_bank.get_single_dataset('MSU_MFSD', 'Train', transform=transformer_train_pure())
msu_loader = torch.utils.data.DataLoader(msu_dataset, batch_size=32, shuffle=False, num_workers=1, pin_memory=True)

replay_dataset = data_bank.get_single_dataset('Replay_attack', 'Train', transform=transformer_train_pure())
replay_loader = torch.utils.data.DataLoader(replay_dataset, batch_size=32, shuffle=False, num_workers=1, pin_memory=True)

zero_samples_1, one_samples_1 = extract_data_labels(casia_loader)
zero_samples_2, one_samples_2 = extract_data_labels(oulu_loader)
zero_samples_3, one_samples_3 = extract_data_labels(msu_loader)
zero_samples_4, one_samples_4 = extract_data_labels(replay_loader)

zero_samples_1, one_samples_1 = model(zero_samples_1), model(one_samples_1)
zero_samples_2, one_samples_2 = model(zero_samples_2), model(one_samples_2)
zero_samples_3, one_samples_3 = model(zero_samples_3), model(one_samples_3)
zero_samples_4, one_samples_4 = model(zero_samples_4), model(one_samples_4)

print('zero_samples_1', zero_samples_1.shape, 'one_samples_1', one_samples_1.shape, 'zero_samples_2', zero_samples_2.shape, 'one_samples_2', one_samples_2.shape, 'zero_samples_3', zero_samples_3.shape, 'one_samples_3', one_samples_3.shape, 'zero_samples_4', zero_samples_4.shape, 'one_samples_4', one_samples_4.shape)

data_ones = torch.cat([one_samples_1, one_samples_2, one_samples_3, one_samples_4], dim=0)
data_zeros = torch.cat([zero_samples_1, zero_samples_2, zero_samples_3, zero_samples_4], dim=0)
label_ones = torch.ones(data_ones.shape[0])
label_zeros = torch.zeros(data_zeros.shape[0])

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ## 按照标签分
# for p in range(5, 35):
    # tsne = TSNE(n_components=2, learning_rate='auto', init='pca')
    # embed_data = torch.cat([data_ones, data_zeros], dim=0)
    # embed_label = torch.cat([torch.ones(data_ones.shape[0]), torch.zeros(data_zeros.shape[0])], dim=0)
    # embed_data = tsne.fit_transform(embed_data.detach().view(embed_data.shape[0], -1).numpy())
    # embedded_data_ones = embed_data[:data_ones.shape[0]]
    # embedded_data_zeros = embed_data[data_ones.shape[0]:]
    # point_size = 10

    # plt.scatter(embedded_data_ones[:, 0], embedded_data_ones[:, 1], c='r', marker='o', s=point_size)
    # plt.scatter(embedded_data_zeros[:, 0], embedded_data_zeros[:, 1], c='g', marker='x', s=point_size)
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # if use_adv:
    #     out = './tsne/label'
    # else:
    #     out = './tsne_not_adv/label'
    # if not os.path.exists(out):
    #     os.makedirs(out)
    # plt.savefig(os.path.join(out, 'label-tsne_{}.png'.format(p)))


## 按照域分
for p in range(5, 50):
    tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=p)
    # print('perplexity', tsne.perplexity)
    # print('zero_samples_1', zero_samples_1.shape, 'one_samples_1', one_samples_1.shape, 'zero_samples_2', zero_samples_2.shape, 'one_samples_2', one_samples_2.shape, 'zero_samples_3', zero_samples_3.shape, 'one_samples_3', one_samples_3.shape, 'zero_samples_4', zero_samples_4.shape, 'one_samples_4', one_samples_4.shape)
    embedded_data_1 = torch.cat([zero_samples_1, one_samples_1], dim=0)
    embedded_data_2 = torch.cat([zero_samples_2, one_samples_2], dim=0)
    embedded_data_3 = torch.cat([zero_samples_3, one_samples_3], dim=0)
    embedded_data_4 = torch.cat([zero_samples_4, one_samples_4], dim=0)
    embed_data = torch.cat([embedded_data_1, embedded_data_2, embedded_data_3, embedded_data_4], dim=0)
    
    embed_data = tsne.fit_transform(embed_data.detach().view(embed_data.shape[0], -1).numpy())
    embedded_data_1 = embed_data[:embedded_data_1.shape[0]]
    embedded_data_2 = embed_data[embedded_data_1.shape[0]:embedded_data_1.shape[0] + embedded_data_2.shape[0]]
    embedded_data_3 = embed_data[embedded_data_1.shape[0] + embedded_data_2.shape[0]:embedded_data_1.shape[0] + embedded_data_2.shape[0] + embedded_data_3.shape[0]]
    embedded_data_4 = embed_data[embedded_data_1.shape[0] + embedded_data_2.shape[0] + embedded_data_3.shape[0]:]
    print(embedded_data_1.shape, embedded_data_2.shape, embedded_data_3.shape, embedded_data_4.shape)
    point_size = 10
    plt.figure()
    num = zero_samples_1.shape[0]
    plt.scatter(embedded_data_1[:num, 0], embedded_data_1[:num, 1], c='r', marker='o', s=point_size, label='target')
    plt.scatter(embedded_data_1[num:, 0], embedded_data_1[num:, 1], c='r', marker='x', s=point_size)
    plt.scatter(embedded_data_2[:num, 0], embedded_data_2[:num, 1], c='g', marker='o', s=point_size, label='domain1')
    plt.scatter(embedded_data_2[num:, 0], embedded_data_2[num:, 1], c='g', marker='x', s=point_size)
    plt.scatter(embedded_data_3[:num, 0], embedded_data_3[:num, 1], c='b', marker='o', s=point_size, label='domain2')
    plt.scatter(embedded_data_3[num:, 0], embedded_data_3[num:, 1], c='b', marker='x', s=point_size)
    plt.scatter(embedded_data_4[:num, 0], embedded_data_4[:num, 1], c='y', marker='o', s=point_size, label='domain3')
    plt.scatter(embedded_data_4[num:, 0], embedded_data_4[num:, 1], c='y', marker='x', s=point_size)
    plt.legend()
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Model With Adversarial Training")
    if use_adv:
        out = './tsne/domain'
    else:
        out = './tsne_not_adv/domain'
    if not os.path.exists(out):
        os.makedirs(out)
    plt.savefig(os.path.join(out, 'domain-tsne_{}.png'.format(p)))
    plt.close()

