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
from glob import glob


frames_total = 8


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


class Spoofing_valtest(Dataset):

    def __init__(self, info_list, jpgs_dir, depth_maps_dir, bboxes_dir, transform=None, img_size=256, map_size=32, UUID=-1):

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

        sample = {'image_x': image_x, 'map_x': map_x, 'label': spoofing_label}
        if self.transform:
            sample = self.transform(sample)
        sample['UUID'] = self.UUID
        return sample
        
    def get_single_image_x(self, jpg_dir_path, map_dir_path, bbox_dir_path, videoname):
        files_total = len([name for name in os.listdir(jpg_dir_path) if os.path.isfile(os.path.join(jpg_dir_path, name))])//3
        interval = files_total//10
        
        image_x = np.zeros((frames_total, 256, 256, 3))
        val_map_x = np.ones((frames_total, 32, 32))
        
        # random choose 1 frame
        image_name = ''
        for i in range(frames_total):
            image_id = i*interval + 1 
            for temp in range(50):
                image_name = "{:03}".format(image_id)
                bbox_path = os.path.join(bbox_dir_path, str(image_name)+'.dat')
                depth_map_path = os.path.join(map_dir_path, str(image_name)+'.jpg')
                
                if os.path.exists(bbox_path) and os.path.exists(depth_map_path)  :    # some scene.dat are missing
                    depth_map_jpg = cv2.imread(depth_map_path, 0)
                    if depth_map_jpg is not None:
                        break
                    else:
                        image_id +=1
                else:
                    image_id +=1
        
            # RGB
            jpg_path = os.path.join(jpg_dir_path, image_name + '.jpg')
            jpg_image = cv2.imread(jpg_path)
            # random scale from [1.2 to 1.5]
            face_scale = np.random.randint(12, 15)
            face_scale = face_scale/10.0
            # gray-map
            depth_map_path = os.path.join(map_dir_path, image_name + '.jpg')
            depth_map_image = cv2.imread(depth_map_path, 0)

            image_x[i,:,:,:] = cv2.resize(crop_face_from_scene(jpg_image, bbox_path, face_scale), (256, 256))
            # transform to binary mask --> threshold = 0 
            depth_image_cropped = cv2.resize(crop_face_from_scene(depth_map_image, bbox_path, face_scale), (32, 32))
            
            np.where(depth_image_cropped < 1, depth_image_cropped, 1)
            val_map_x[i,:,:] = depth_image_cropped
            
			
        return image_x, val_map_x