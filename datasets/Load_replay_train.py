import os
import torch
import pandas as pd
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
import math
import os 
from utils import *
from glob import glob
import imgaug.augmenters as iaa

seq = iaa.Sequential([
    iaa.Add(value=(-40,40), per_channel=True), # Add color 
    iaa.GammaContrast(gamma=(0.5,1.5)) # GammaContrast with a gamma of 0.5 to 1.5
])

def crop_face_from_scene(image,face_name_full, scale):
    f=open(face_name_full,'r')
    lines=f.readlines()
    y1,x1,w,h=[float(ele) for ele in lines[:4]]
    f.close()
    y2=y1+w
    x2=x1+h
    y_mid=(y1+y2)/2.0
    x_mid=(x1+x2)/2.0
    h_img, w_img = image.shape[0], image.shape[1]
    w_scale=scale*w
    h_scale=scale*h
    y1=y_mid-w_scale/2.0
    x1=x_mid-h_scale/2.0
    y2=y_mid+w_scale/2.0
    x2=x_mid+h_scale/2.0
    y1=max(math.floor(y1),0)
    x1=max(math.floor(x1),0)
    y2=min(math.floor(y2),w_img)
    x2=min(math.floor(x2),h_img)
    region=image[x1:x2,y1:y2]
    return region


class Spoofing_train(Dataset):
    
    def __init__(self, info_list, jpgs_dir, depth_maps_dir, bboxes_dir, transform=None, scale_up=1.5, scale_down=1.0, img_size=256, map_size=32, UUID=-1):
        self.landmarks_frame = pd.read_csv(info_list, delimiter=',', header=None)
        self.jpgs_dir = jpgs_dir
        self.depth_maps_dir = depth_maps_dir
        self.bboxes_dir = bboxes_dir
        self.transform = transform
        self.UUID = UUID

    def __len__(self):
        return len(self.landmarks_frame)
    
    def __getitem__(self, idx):
        video_name = str(self.landmarks_frame.iloc[idx, 1])
        jpgs_path = os.path.join(self.jpgs_dir, video_name)
        depth_maps_path = os.path.join(self.depth_maps_dir, video_name)
        bboxs_path = os.path.join(self.bboxes_dir, video_name)
             
        image_x, map_x = self.get_single_image_x(jpgs_path, depth_maps_path, bboxs_path, video_name)

        spoofing_label = self.landmarks_frame.iloc[idx, 0]
        if spoofing_label == 1:
            spoofing_label = 1            # real
        else:
            spoofing_label = 0
            map_x = np.zeros((32, 32))    # fake

        sample = {'image_x': image_x, 'map_x': map_x, 'label': spoofing_label, "UUID": self.UUID}
        if self.transform:
            sample = self.transform(sample)
        sample['UUID'] = self.UUID
        return sample
    
    # jpg_dir_path: /root/autodl-tmp/oulu_depth/Dev_jpgs/6_3_22_4
    # map_dir_path: /root/autodl-tmp/oulu_depth/Dev_depth/6_3_22_4
    # bbox_dir_path: /root/autodl-tmp/oulu_depth/Dev_bbox/6_3_22_4
    def get_single_image_x(self, jpg_dir_path, map_dir_path, bbox_dir_path, video_name):

        frames_total = len([name for name in os.listdir(map_dir_path) if os.path.isfile(os.path.join(map_dir_path, name))])
        # random choose 1 frame
        image_name = ''
        for i in range(500):
            image_id = np.random.randint(1, frames_total-1)
            image_name = "{:03}".format(image_id)
            bbox_path = os.path.join(bbox_dir_path, str(image_name)+'.dat')
            depth_map_path = os.path.join(map_dir_path, str(image_name)+'.jpg')
        
            # some .dat & map files have been missing  
            if os.path.exists(depth_map_path) and os.path.exists(bbox_path):
                depth_map_jpg = cv2.imread(depth_map_path, 0)
                if depth_map_jpg is not None:
                    break
        
        if image_name == '':
            print('error: ', depth_map_path, bbox_path)
            return np.zeros((256, 256, 3)), np.zeros((32, 32))
        
        # random scale from [1.2 to 1.5]
        face_scale = np.random.randint(12, 15)
        face_scale = face_scale/10.0
        
        # RGB
        jpg_path = os.path.join(jpg_dir_path, image_name + '.jpg')
        jpg_image = cv2.imread(jpg_path)

        # gray-map
        depth_map_path = os.path.join(map_dir_path, image_name + '.jpg')
        depth_map_image = cv2.imread(depth_map_path, 0)
        bbox_path = os.path.join(bbox_dir_path, image_name + '.dat')

        face_image_cropped = cv2.resize(crop_face_from_scene(jpg_image, bbox_path, face_scale), (256, 256))
        # data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
        # image_x_aug = seq.augment_image(face_image_cropped) 
<<<<<<< HEAD
        image_x_aug = face_image_cropped
=======
>>>>>>> feat: update

        depth_image_cropped = cv2.resize(crop_face_from_scene(depth_map_image, bbox_path, face_scale), (32, 32))
    
        return face_image_cropped, depth_image_cropped