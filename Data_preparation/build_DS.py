import os
import time
import uuid
import cv2
import xlsxwriter
import pandas as pd

IMAGES_PATH = 'E:/J#4/Data/Input'
IMAGES_PATH1 = 'E:/J#4/Test/Input'
DS_PATH='E:/J#4/Data/Preprocessing/DS'
DS_PATH1='E:/J#4/Test/Preprocessing/DS'


import tensorflow as tf
import cv2
import json
import numpy as np
from matplotlib import pyplot as plt
import math

from random import uniform, seed
import random

import albumentations as alb
augmentor = alb.Compose([alb.RandomCrop(width=450, height=450), 
                            alb.ShiftScaleRotate(shift_limit=0.1,  rotate_limit=10, border_mode=0, p=0.5), 
                            alb.HorizontalFlip(p=0.5), 
                            alb.RandomBrightnessContrast(p=0.2),
                            alb.RandomGamma(p=0.2), 
                            alb.RGBShift(p=0.2), 
                            alb.RandomScale(scale_limit=(-0.5, 0.5), p=1, interpolation=1),
                            alb.PadIfNeeded(450, 450, border_mode=cv2.BORDER_CONSTANT), 
                            alb.Resize(450, 450, cv2.INTER_NEAREST)], 
                            keypoint_params=alb.KeypointParams(format='xy', label_fields=['class_labels']))

augmentor1 = alb.Compose([alb.Resize(height=250, width=250, p=1)],
                        keypoint_params=alb.KeypointParams(format='xy', label_fields=['class_labels']))

def shifted_landmarks(correct_coords,w_,h_, path_, name_):
    
    incorrect_coords=np.zeros(36,dtype=float)
    for f in range(len(correct_coords)):
        incorrect_coords[f]=correct_coords[f]
    
    count_shifted_landmark = uniform(3, 8)
    
    num_lst = []
    x_lst=[]
    y_lst=[]
    x_position=[]
    y_position=[]
    num_landmark = random.sample(range(1, 19), int(count_shifted_landmark))
    
    for landmark_counter in range(int(count_shifted_landmark)):
        x_val = uniform(1, 6) 
        x_pos=random.randrange(1, 3, 1)
        x_lst.append(int(x_val))
        x_position.append(int(x_pos)) 
        y_val = uniform(1, 6) 
        y_pos=random.randrange(1, 3, 1)
        y_lst.append(int(y_val))
        y_position.append(int(y_pos)) 

    for kc in range(len(num_landmark)):
        pos=num_landmark[kc]
        x_index=2*pos-2
        y_index=2*pos-1
        
        if x_position[kc]==1:
            if((correct_coords[x_index]-x_lst[kc])>0):
                incorrect_coords[x_index]=correct_coords[x_index]-x_lst[kc]
            else:
                incorrect_coords[x_index]=correct_coords[x_index]+x_lst[kc] 
        else:
            if((correct_coords[x_index]+x_lst[kc])>w_):
                incorrect_coords[x_index]=correct_coords[x_index]-x_lst[kc]
            else:
                incorrect_coords[x_index]=correct_coords[x_index]+x_lst[kc] 

        if y_position[kc]==1:
            if((correct_coords[y_index]-y_lst[kc])>0):
                incorrect_coords[y_index]=correct_coords[y_index]-y_lst[kc]
            else:
                incorrect_coords[y_index]=correct_coords[y_index]+y_lst[kc]
        else:
            if((correct_coords[y_index]+y_lst[kc])>h_):
                incorrect_coords[y_index]=correct_coords[y_index]-y_lst[kc]
            else:
                incorrect_coords[y_index]=correct_coords[y_index]+y_lst[kc]
            
    w1=w_
    h1=h_
    
    annotation = {}
    annotation['image'] = name_
    annotation['keypoints'] = np.zeros(36,dtype=float)
    annotation['keypoints'][0] = incorrect_coords[0]/w1
    annotation['keypoints'][1] = incorrect_coords[1]/h1
    annotation['keypoints'][2] = incorrect_coords[2]/w1
    annotation['keypoints'][3] = incorrect_coords[3]/h1
    annotation['keypoints'][4] = incorrect_coords[4]/w1
    annotation['keypoints'][5] = incorrect_coords[5]/h1
    annotation['keypoints'][6] = incorrect_coords[6]/w1
    annotation['keypoints'][7] = incorrect_coords[7]/h1
    annotation['keypoints'][8] = incorrect_coords[8]/w1
    annotation['keypoints'][9] = incorrect_coords[9]/h1
    annotation['keypoints'][10] = incorrect_coords[10]/w1
    annotation['keypoints'][11] = incorrect_coords[11]/h1
    annotation['keypoints'][12] = incorrect_coords[12]/w1
    annotation['keypoints'][13] = incorrect_coords[13]/h1
    annotation['keypoints'][14] = incorrect_coords[14]/w1
    annotation['keypoints'][15] = incorrect_coords[15]/h1
    annotation['keypoints'][16] = incorrect_coords[16]/w1
    annotation['keypoints'][17] = incorrect_coords[17]/h1
    annotation['keypoints'][18] = incorrect_coords[18]/w1
    annotation['keypoints'][19] = incorrect_coords[19]/h1
    annotation['keypoints'][20] = incorrect_coords[20]/w1
    annotation['keypoints'][21] = incorrect_coords[21]/h1
    annotation['keypoints'][22] = incorrect_coords[22]/w1
    annotation['keypoints'][23] = incorrect_coords[23]/h1
    annotation['keypoints'][24] = incorrect_coords[24]/w1
    annotation['keypoints'][25] = incorrect_coords[25]/h1
    annotation['keypoints'][26] = incorrect_coords[26]/w1
    annotation['keypoints'][27] = incorrect_coords[27]/h1
    annotation['keypoints'][28] = incorrect_coords[28]/w1
    annotation['keypoints'][29] = incorrect_coords[29]/h1
    annotation['keypoints'][30] = incorrect_coords[30]/w1
    annotation['keypoints'][31] = incorrect_coords[31]/h1
    annotation['keypoints'][32] = incorrect_coords[32]/w1
    annotation['keypoints'][33] = incorrect_coords[33]/h1
    annotation['keypoints'][34] = incorrect_coords[34]/w1  
    annotation['keypoints'][35] = incorrect_coords[35]/h1

    matrix=np.ones(36,dtype=float)        
    annotation['keypoints'] = list(np.divide(annotation['keypoints'], matrix))
    with open(path_, 'w') as f: 
        json.dump(annotation, f)

def add_noise(img,path_):
    
    gauss = np.random.normal(0,1,img.size)
    gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
    img_gauss = cv2.add(img,gauss)
    cv2.imwrite(path_, img_gauss)

def Prepare_Original_DS():
    ### Step of work: 1-crop (_origDS_crop) 2-resize (250*250) (_origDS_crop_resize)
    rows=0
    df_crop_test = pd.DataFrame(columns=['img_name','x1_coord','y1_coord', 'x2_coord','y2_coord'])
    coords_rec=np.zeros(4,dtype=int)

    for partition in ['train', 'test', 'val']: 
        for image in os.listdir(os.path.join(IMAGES_PATH, partition, 'images')):
            img = cv2.imread(os.path.join(IMAGES_PATH, partition, 'images', image))
            img_name=f'{image.split(".")[0]}'
            #print(img_name,"   *****")
            coords=np.zeros(36,dtype=float)
            label_path = os.path.join(IMAGES_PATH, partition, 'labels', f'{image.split(".")[0]}.json')
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    label = json.load(f)

                if len(label['shapes']) > 1:     
                                        
                    if label['shapes'][2]['label']=='NtL': 
                        coords[0] = np.squeeze(label['shapes'][2]['points'])[0]
                        coords[1] = np.squeeze(label['shapes'][2]['points'])[1]

                    if label['shapes'][3]['label']=='NtR':
                        coords[2] = np.squeeze(label['shapes'][3]['points'])[0]
                        coords[3] = np.squeeze(label['shapes'][3]['points'])[1]

                    if label['shapes'][4]['label']=='NbL': 
                        coords[4] = np.squeeze(label['shapes'][4]['points'])[0]
                        coords[5] = np.squeeze(label['shapes'][4]['points'])[1]

                    if label['shapes'][5]['label']=='NbR':
                        coords[6] = np.squeeze(label['shapes'][5]['points'])[0]
                        coords[7] = np.squeeze(label['shapes'][5]['points'])[1]

                    if label['shapes'][6]['label']=='NlL': 
                        coords[8] = np.squeeze(label['shapes'][6]['points'])[0]
                        coords[9] = np.squeeze(label['shapes'][6]['points'])[1]

                    if label['shapes'][7]['label']=='NlR':
                        coords[10] = np.squeeze(label['shapes'][7]['points'])[0]
                        coords[11] = np.squeeze(label['shapes'][7]['points'])[1]

                    if label['shapes'][8]['label']=='NmL': 
                        coords[12] = np.squeeze(label['shapes'][8]['points'])[0]
                        coords[13] = np.squeeze(label['shapes'][8]['points'])[1]

                    if label['shapes'][9]['label']=='NmR':
                        coords[14] = np.squeeze(label['shapes'][9]['points'])[0]
                        coords[15] = np.squeeze(label['shapes'][9]['points'])[1]

                    if label['shapes'][10]['label'] =='AlL': 
                        coords[16]= np.squeeze(label['shapes'][10]['points'])[0]
                        coords[17] = np.squeeze(label['shapes'][10]['points'])[1]

                    if label['shapes'][11]['label'] =='AlR': 
                        coords[18] = np.squeeze(label['shapes'][11]['points'])[0]
                        coords[19] = np.squeeze(label['shapes'][11]['points'])[1]
                    
                    if label['shapes'][12]['label']=='SbalL': 
                        coords[20] = np.squeeze(label['shapes'][12]['points'])[0]
                        coords[21] = np.squeeze(label['shapes'][12]['points'])[1]

                    if label['shapes'][13]['label']=='SbalR':
                        coords[22] = np.squeeze(label['shapes'][13]['points'])[0]
                        coords[23] = np.squeeze(label['shapes'][13]['points'])[1]

                    if label['shapes'][14]['label']=='Alt1L': 
                        coords[24] = np.squeeze(label['shapes'][14]['points'])[0]
                        coords[25] = np.squeeze(label['shapes'][14]['points'])[1]

                    if label['shapes'][15]['label']=='Alt1R':
                        coords[26] = np.squeeze(label['shapes'][15]['points'])[0]
                        coords[27] = np.squeeze(label['shapes'][15]['points'])[1]

                    if label['shapes'][16]['label']=='Alt2L': 
                        coords[28] = np.squeeze(label['shapes'][16]['points'])[0]
                        coords[29] = np.squeeze(label['shapes'][16]['points'])[1]

                    if label['shapes'][17]['label']=='Alt2R':
                        coords[30] = np.squeeze(label['shapes'][17]['points'])[0]
                        coords[31] = np.squeeze(label['shapes'][17]['points'])[1]

                    if label['shapes'][18]['label']=='Prn': 
                        coords[32] = np.squeeze(label['shapes'][18]['points'])[0]
                        coords[33] = np.squeeze(label['shapes'][18]['points'])[1]

                    if label['shapes'][19]['label']=='Sn':
                        coords[34] = np.squeeze(label['shapes'][19]['points'])[0]
                        coords[35] = np.squeeze(label['shapes'][19]['points'])[1]
   
                matrix=np.ones(36,dtype=float) 
                for i1 in range(0,36):
                    matrix[i1]=450 
                """
                if partition=='test': 
                    print(coords,"**********************")    
                    print(np.divide(coords, matrix))
                """
                ########################### Resize Original Images
                matrix=np.ones(36,dtype=float)                    
                
                keypoints1 = [(coords[:2]), (coords[2:4]), (coords[4:6]), (coords[6:8]),(coords[8:10]), 
                            (coords[10:12]),(coords[12:14]), (coords[14:16]), (coords[16:18]), (coords[18:20]),
                            (coords[20:22]),(coords[22:24]), (coords[24:26]), (coords[26:28]), (coords[28:30]),
                            (coords[30:32]),(coords[32:34]), (coords[34:36])]
                    
                augmented1 = augmentor1(image=img, keypoints=keypoints1, class_labels=['NtL','NtR','NbL','NbR','NlL','NlR','NmL','NmR','AlL','AlR',
                                                                                            'SbalL','SbalR','Alt1L','Alt1R','Alt2L','Alt2R','Prn','Sn'])

                cv2.imwrite(os.path.join(DS_PATH,'OrigDS', partition, 'Real','images_resize', f'{image.split(".")[0]}.jpg'), augmented1['image'])              
                #image1=f'{image.split(".")[0]}.jpg'
                annotation1 = {}
                annotation1['image'] = image
                annotation1['keypoints'] = np.zeros(36,dtype=float)
                resize_coords=np.zeros(36,dtype=float)  
                matrix=np.ones(36, dtype=int)
      
                if len(augmented1['keypoints']) > 0: 
                    for idx, cl in enumerate(augmented1['class_labels']):
                        if cl == 'NtL': 
                            annotation1['keypoints'][0] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][1] = augmented1['keypoints'][idx][1]
                            resize_coords[0]=augmented1['keypoints'][idx][0]
                            resize_coords[1]=augmented1['keypoints'][idx][1]
                        if cl == 'NtR': 
                            annotation1['keypoints'][2] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][3] = augmented1['keypoints'][idx][1]
                            resize_coords[2]=augmented1['keypoints'][idx][0]
                            resize_coords[3]=augmented1['keypoints'][idx][1]
                        if cl == 'NbL':
                            annotation1['keypoints'][4] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][5] = augmented1['keypoints'][idx][1]
                            resize_coords[4]=augmented1['keypoints'][idx][0]
                            resize_coords[5]=augmented1['keypoints'][idx][1]
                        if cl == 'NbR':  
                            annotation1['keypoints'][6] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][7] = augmented1['keypoints'][idx][1]
                            resize_coords[6]=augmented1['keypoints'][idx][0]
                            resize_coords[7]=augmented1['keypoints'][idx][1]
                        if cl == 'NlL': 
                            annotation1['keypoints'][8] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][9] = augmented1['keypoints'][idx][1]
                            resize_coords[8]=augmented1['keypoints'][idx][0]
                            resize_coords[9]=augmented1['keypoints'][idx][1]
                        if cl == 'NlR': 
                            annotation1['keypoints'][10] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][11] = augmented1['keypoints'][idx][1]
                            resize_coords[10]=augmented1['keypoints'][idx][0]
                            resize_coords[11]=augmented1['keypoints'][idx][1]
                        if cl == 'NmL': 
                            annotation1['keypoints'][12] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][13] = augmented1['keypoints'][idx][1]
                            resize_coords[12]=augmented1['keypoints'][idx][0]
                            resize_coords[13]=augmented1['keypoints'][idx][1]
                        if cl == 'NmR': 
                            annotation1['keypoints'][14] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][15] = augmented1['keypoints'][idx][1]
                            resize_coords[14]=augmented1['keypoints'][idx][0]
                            resize_coords[15]=augmented1['keypoints'][idx][1]
                        if cl == 'AlL': 
                            annotation1['keypoints'][16] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][17] = augmented1['keypoints'][idx][1]
                            resize_coords[16]=augmented1['keypoints'][idx][0]
                            resize_coords[17]=augmented1['keypoints'][idx][1]
                        if cl == 'AlR':
                            annotation1['keypoints'][18] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][19] = augmented1['keypoints'][idx][1]
                            resize_coords[18]=augmented1['keypoints'][idx][0]
                            resize_coords[19]=augmented1['keypoints'][idx][1]
                        if cl == 'SbalL':
                            annotation1['keypoints'][20] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][21] = augmented1['keypoints'][idx][1]
                            resize_coords[20]=augmented1['keypoints'][idx][0]
                            resize_coords[21]=augmented1['keypoints'][idx][1]
                        if cl == 'SbalR':  
                            annotation1['keypoints'][22] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][23] = augmented1['keypoints'][idx][1]
                            resize_coords[22]=augmented1['keypoints'][idx][0]
                            resize_coords[23]=augmented1['keypoints'][idx][1]
                        if cl == 'Alt1L': 
                            annotation1['keypoints'][24] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][25] = augmented1['keypoints'][idx][1]
                            resize_coords[24]=augmented1['keypoints'][idx][0]
                            resize_coords[25]=augmented1['keypoints'][idx][1]
                        if cl == 'Alt1R': 
                            annotation1['keypoints'][26] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][27] = augmented1['keypoints'][idx][1]
                            resize_coords[26]=augmented1['keypoints'][idx][0]
                            resize_coords[27]=augmented1['keypoints'][idx][1]
                        if cl == 'Alt2L':
                            annotation1['keypoints'][28] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][29] = augmented1['keypoints'][idx][1]
                            resize_coords[28]=augmented1['keypoints'][idx][0]
                            resize_coords[29]=augmented1['keypoints'][idx][1]
                        if cl == 'Alt2R':
                            annotation1['keypoints'][30] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][31] = augmented1['keypoints'][idx][1]
                            resize_coords[30]=augmented1['keypoints'][idx][0]
                            resize_coords[31]=augmented1['keypoints'][idx][1]
                        if cl == 'Prn':  
                            annotation1['keypoints'][32] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][33] = augmented1['keypoints'][idx][1]
                            resize_coords[32]=augmented1['keypoints'][idx][0]
                            resize_coords[33]=augmented1['keypoints'][idx][1]
                        if cl == 'Sn': 
                            annotation1['keypoints'][34] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][35] = augmented1['keypoints'][idx][1]
                            resize_coords[34]=augmented1['keypoints'][idx][0]
                            resize_coords[35]=augmented1['keypoints'][idx][1]
                        
                for i1 in range(0,36):
                    matrix[i1]=250
                
                annotation1['keypoints'] = list(np.divide(annotation1['keypoints'], matrix))
                
                with open(os.path.join(DS_PATH,'OrigDS', partition, 'Real','labels_resize', f'{image.split(".")[0]}.json'), 'w') as f1: 
                    json.dump(annotation1, f1)  
                
                file_path=  os.path.join(DS_PATH,'OrigDS', partition, 'Shift','labels_resize', f'{image.split(".")[0]}.json')                 
                shifted_landmarks(resize_coords,250,250,file_path,annotation1['image'])     
                
                file_path_= os.path.join(DS_PATH,'OrigDS', partition, 'Noise','images_resize', f'{image.split(".")[0]}.jpg')
                add_noise(augmented1['image'],file_path_)   
                resize_coords=np.zeros(36,dtype=float) 
                
                ######################## Crop Original Images               
                crop_coords=np.zeros(36,dtype=float)
                matrix=np.ones(36,dtype=float) 
                               
                x_coords=[]
                y_coords=[]
                for f in range(len(coords)):
                    if f%2==0: x_coords.append(round(coords[f],4))
                    else: y_coords.append(round(coords[f],4))
                x_min=min(x_coords)
                x_max=max(x_coords)
                y_min=min(y_coords)
                y_max=max(y_coords)

                y1_crop=int(y_min-9)
                y2_crop=int(y_max+15)
                x1_crop=int(x_min-9)
                x2_crop=int(x_max+9)

                if partition=='test':
                    df_crop_test = df_crop_test.append({'img_name':image,'x1_coord':x1_crop,'y1_coord':y1_crop, 'x2_coord':x2_crop,'y2_coord':y2_crop}, ignore_index = True)

                w1=x2_crop-x1_crop
                h1=y2_crop-y1_crop
                crop_img=img[y1_crop:y2_crop, x1_crop:x2_crop]           
                cv2.imwrite(os.path.join(DS_PATH,'OrigDS', partition, 'Real','images_crop', f'{image.split(".")[0]}.jpg'), crop_img)
                                
                for i_ in range (len(coords)):
                    if i_%2==0: crop_coords[i_]=(coords[i_]-x1_crop)#/w1
                    else: crop_coords[i_]=(coords[i_]-y1_crop)#/h1
                annotation = {}
                annotation['image'] = image
                annotation['keypoints'] = np.zeros(36,dtype=float)
                annotation['keypoints'][0] = crop_coords[0]/w1
                annotation['keypoints'][1] = crop_coords[1]/h1
                annotation['keypoints'][2] = crop_coords[2]/w1
                annotation['keypoints'][3] = crop_coords[3]/h1
                annotation['keypoints'][4] = crop_coords[4]/w1
                annotation['keypoints'][5] = crop_coords[5]/h1
                annotation['keypoints'][6] = crop_coords[6]/w1
                annotation['keypoints'][7] = crop_coords[7]/h1
                annotation['keypoints'][8] = crop_coords[8]/w1
                annotation['keypoints'][9] = crop_coords[9]/h1
                annotation['keypoints'][10] = crop_coords[10]/w1
                annotation['keypoints'][11] = crop_coords[11]/h1
                annotation['keypoints'][12] = crop_coords[12]/w1
                annotation['keypoints'][13] = crop_coords[13]/h1
                annotation['keypoints'][14] = crop_coords[14]/w1
                annotation['keypoints'][15] = crop_coords[15]/h1
                annotation['keypoints'][16] = crop_coords[16]/w1
                annotation['keypoints'][17] = crop_coords[17]/h1
                annotation['keypoints'][18] = crop_coords[18]/w1
                annotation['keypoints'][19] = crop_coords[19]/h1
                annotation['keypoints'][20] = crop_coords[20]/w1
                annotation['keypoints'][21] = crop_coords[21]/h1
                annotation['keypoints'][22] = crop_coords[22]/w1
                annotation['keypoints'][23] = crop_coords[23]/h1
                annotation['keypoints'][24] = crop_coords[24]/w1
                annotation['keypoints'][25] = crop_coords[25]/h1
                annotation['keypoints'][26] = crop_coords[26]/w1
                annotation['keypoints'][27] = crop_coords[27]/h1
                annotation['keypoints'][28] = crop_coords[28]/w1
                annotation['keypoints'][29] = crop_coords[29]/h1
                annotation['keypoints'][30] = crop_coords[30]/w1
                annotation['keypoints'][31] = crop_coords[31]/h1
                annotation['keypoints'][32] = crop_coords[32]/w1
                annotation['keypoints'][33] = crop_coords[33]/h1
                annotation['keypoints'][34] = crop_coords[34]/w1  
                annotation['keypoints'][35] = crop_coords[35]/h1
                          
                annotation['keypoints'] = list(np.divide(annotation['keypoints'], matrix))
                with open(os.path.join(DS_PATH,'OrigDS', partition, 'Real','labels_crop', f'{image.split(".")[0]}.json'), 'w') as f: 
                    json.dump(annotation, f)
                
                file_path=  os.path.join(DS_PATH,'OrigDS', partition, 'Shift','labels_crop', f'{image.split(".")[0]}.json')                 
                shifted_landmarks(crop_coords,w1,h1,file_path,annotation['image']) 
                
                file_path_= os.path.join(DS_PATH,'OrigDS', partition, 'Noise','images_crop', f'{image.split(".")[0]}.jpg')
                add_noise(crop_img,file_path_) 
                ##################################### Resize Crop Images
                keypoints1 = [(crop_coords[:2]), (crop_coords[2:4]), (crop_coords[4:6]), (crop_coords[6:8]),(crop_coords[8:10]), 
                    (crop_coords[10:12]),(crop_coords[12:14]), (crop_coords[14:16]), (crop_coords[16:18]), (crop_coords[18:20]),
                    (crop_coords[20:22]),(crop_coords[22:24]), (crop_coords[24:26]), (crop_coords[26:28]), (crop_coords[28:30]),
                    (crop_coords[30:32]),(crop_coords[32:34]), (crop_coords[34:36])]
            
                augmented1 = augmentor1(image=crop_img, keypoints=keypoints1, class_labels=['NtL','NtR','NbL','NbR','NlL','NlR','NmL','NmR','AlL','AlR',
                                                                                            'SbalL','SbalR','Alt1L','Alt1R','Alt2L','Alt2R','Prn','Sn'])

                cv2.imwrite(os.path.join(DS_PATH,'OrigDS', partition, 'Real','images_crop_resize', f'{image.split(".")[0]}.jpg'), augmented1['image'])
                
                resized_img=augmented1['image']
                #image1=f'{image.split(".")[0]}.jpg'
                annotation1 = {}
                annotation1['image'] = image
                annotation1['keypoints'] = np.zeros(36,dtype=float)
                resize_coords=np.zeros(36,dtype=float)  
                    
                if len(augmented1['keypoints']) > 0: 
                    for idx, cl in enumerate(augmented1['class_labels']):
                        if cl == 'NtL': 
                            annotation1['keypoints'][0] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][1] = augmented1['keypoints'][idx][1]
                            resize_coords[0]=augmented1['keypoints'][idx][0]
                            resize_coords[1]=augmented1['keypoints'][idx][1]
                        if cl == 'NtR': 
                            annotation1['keypoints'][2] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][3] = augmented1['keypoints'][idx][1]
                            resize_coords[2]=augmented1['keypoints'][idx][0]
                            resize_coords[3]=augmented1['keypoints'][idx][1]
                        if cl == 'NbL':
                            annotation1['keypoints'][4] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][5] = augmented1['keypoints'][idx][1]
                            resize_coords[4]=augmented1['keypoints'][idx][0]
                            resize_coords[5]=augmented1['keypoints'][idx][1]
                        if cl == 'NbR':  
                            annotation1['keypoints'][6] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][7] = augmented1['keypoints'][idx][1]
                            resize_coords[6]=augmented1['keypoints'][idx][0]
                            resize_coords[7]=augmented1['keypoints'][idx][1]
                        if cl == 'NlL': 
                            annotation1['keypoints'][8] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][9] = augmented1['keypoints'][idx][1]
                            resize_coords[8]=augmented1['keypoints'][idx][0]
                            resize_coords[9]=augmented1['keypoints'][idx][1]
                        if cl == 'NlR': 
                            annotation1['keypoints'][10] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][11] = augmented1['keypoints'][idx][1]
                            resize_coords[10]=augmented1['keypoints'][idx][0]
                            resize_coords[11]=augmented1['keypoints'][idx][1]
                        if cl == 'NmL': 
                            annotation1['keypoints'][12] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][13] = augmented1['keypoints'][idx][1]
                            resize_coords[12]=augmented1['keypoints'][idx][0]
                            resize_coords[13]=augmented1['keypoints'][idx][1]
                        if cl == 'NmR': 
                            annotation1['keypoints'][14] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][15] = augmented1['keypoints'][idx][1]
                            resize_coords[14]=augmented1['keypoints'][idx][0]
                            resize_coords[15]=augmented1['keypoints'][idx][1]
                        if cl == 'AlL': 
                            annotation1['keypoints'][16] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][17] = augmented1['keypoints'][idx][1]
                            resize_coords[16]=augmented1['keypoints'][idx][0]
                            resize_coords[17]=augmented1['keypoints'][idx][1]
                        if cl == 'AlR':
                            annotation1['keypoints'][18] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][19] = augmented1['keypoints'][idx][1]
                            resize_coords[18]=augmented1['keypoints'][idx][0]
                            resize_coords[19]=augmented1['keypoints'][idx][1]
                        if cl == 'SbalL':
                            annotation1['keypoints'][20] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][21] = augmented1['keypoints'][idx][1]
                            resize_coords[20]=augmented1['keypoints'][idx][0]
                            resize_coords[21]=augmented1['keypoints'][idx][1]
                        if cl == 'SbalR':  
                            annotation1['keypoints'][22] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][23] = augmented1['keypoints'][idx][1]
                            resize_coords[22]=augmented1['keypoints'][idx][0]
                            resize_coords[23]=augmented1['keypoints'][idx][1]
                        if cl == 'Alt1L': 
                            annotation1['keypoints'][24] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][25] = augmented1['keypoints'][idx][1]
                            resize_coords[24]=augmented1['keypoints'][idx][0]
                            resize_coords[25]=augmented1['keypoints'][idx][1]
                        if cl == 'Alt1R': 
                            annotation1['keypoints'][26] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][27] = augmented1['keypoints'][idx][1]
                            resize_coords[26]=augmented1['keypoints'][idx][0]
                            resize_coords[27]=augmented1['keypoints'][idx][1]
                        if cl == 'Alt2L':
                            annotation1['keypoints'][28] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][29] = augmented1['keypoints'][idx][1]
                            resize_coords[28]=augmented1['keypoints'][idx][0]
                            resize_coords[29]=augmented1['keypoints'][idx][1]
                        if cl == 'Alt2R':
                            annotation1['keypoints'][30] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][31] = augmented1['keypoints'][idx][1]
                            resize_coords[30]=augmented1['keypoints'][idx][0]
                            resize_coords[31]=augmented1['keypoints'][idx][1]
                        if cl == 'Prn':  
                            annotation1['keypoints'][32] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][33] = augmented1['keypoints'][idx][1]
                            resize_coords[32]=augmented1['keypoints'][idx][0]
                            resize_coords[33]=augmented1['keypoints'][idx][1]
                        if cl == 'Sn': 
                            annotation1['keypoints'][34] = augmented1['keypoints'][idx][0]
                            annotation1['keypoints'][35] = augmented1['keypoints'][idx][1]
                            resize_coords[34]=augmented1['keypoints'][idx][0]
                            resize_coords[35]=augmented1['keypoints'][idx][1]
                                            
                for i_ in range(36):
                    matrix[i_]=250 

                annotation1['keypoints'] = list(np.divide(annotation1['keypoints'], matrix))

                with open(os.path.join(DS_PATH,'OrigDS', partition,'Real', 'labels_crop_resize', f'{image.split(".")[0]}.json'), 'w') as f1: 
                    json.dump(annotation1, f1)   
                
                file_path=  os.path.join(DS_PATH,'OrigDS', partition, 'Shift','labels_crop_resize', f'{image.split(".")[0]}.json')                 
                shifted_landmarks(resize_coords,250,250,file_path,annotation1['image'])  
                
                file_path_= os.path.join(DS_PATH,'OrigDS', partition, 'Noise','images_crop_resize', f'{image.split(".")[0]}.jpg')
                add_noise(resized_img,file_path_)    
               
            print(rows,": partition: ", partition, f'{image.split(".")[0]}.jpg') 
            rows=rows+1
    writer = pd.ExcelWriter('E:/J#4/Data/Output/crop_coords_test.xlsx')
    df_crop_test.to_excel(writer)
    writer.save()
            
def Prepare_Augmented_DS():
    ### Step of work: 1-Augment Original Images 2- Resize Augmented Images (250*250) 3-Crop Augmented Images 4-Resize Cropped Images
    rows=0
    for partition in ['train']: 
        
        for image in os.listdir(os.path.join(IMAGES_PATH, partition, 'images')):
            
            img = cv2.imread(os.path.join(IMAGES_PATH, partition, 'images', image))
            
            coords=np.zeros(36,dtype=float)
            label_path = os.path.join(IMAGES_PATH, partition, 'labels', f'{image.split(".")[0]}.json')
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    label = json.load(f)
        
                if len(label['shapes']) > 1:     
                    if label['shapes'][2]['label']=='NtL': 
                        coords[0] = np.squeeze(label['shapes'][2]['points'])[0]
                        coords[1] = np.squeeze(label['shapes'][2]['points'])[1]

                    if label['shapes'][3]['label']=='NtR':
                        coords[2] = np.squeeze(label['shapes'][3]['points'])[0]
                        coords[3] = np.squeeze(label['shapes'][3]['points'])[1]

                    if label['shapes'][4]['label']=='NbL': 
                        coords[4] = np.squeeze(label['shapes'][4]['points'])[0]
                        coords[5] = np.squeeze(label['shapes'][4]['points'])[1]

                    if label['shapes'][5]['label']=='NbR':
                        coords[6] = np.squeeze(label['shapes'][5]['points'])[0]
                        coords[7] = np.squeeze(label['shapes'][5]['points'])[1]

                    if label['shapes'][6]['label']=='NlL': 
                        coords[8] = np.squeeze(label['shapes'][6]['points'])[0]
                        coords[9] = np.squeeze(label['shapes'][6]['points'])[1]

                    if label['shapes'][7]['label']=='NlR':
                        coords[10] = np.squeeze(label['shapes'][7]['points'])[0]
                        coords[11] = np.squeeze(label['shapes'][7]['points'])[1]

                    if label['shapes'][8]['label']=='NmL': 
                        coords[12] = np.squeeze(label['shapes'][8]['points'])[0]
                        coords[13] = np.squeeze(label['shapes'][8]['points'])[1]

                    if label['shapes'][9]['label']=='NmR':
                        coords[14] = np.squeeze(label['shapes'][9]['points'])[0]
                        coords[15] = np.squeeze(label['shapes'][9]['points'])[1]

                    if label['shapes'][10]['label'] =='AlL': 
                        coords[16]= np.squeeze(label['shapes'][10]['points'])[0]
                        coords[17] = np.squeeze(label['shapes'][10]['points'])[1]

                    if label['shapes'][11]['label'] =='AlR': 
                        coords[18] = np.squeeze(label['shapes'][11]['points'])[0]
                        coords[19] = np.squeeze(label['shapes'][11]['points'])[1]
                        
                    if label['shapes'][12]['label']=='SbalL': 
                        coords[20] = np.squeeze(label['shapes'][12]['points'])[0]
                        coords[21] = np.squeeze(label['shapes'][12]['points'])[1]

                    if label['shapes'][13]['label']=='SbalR':
                        coords[22] = np.squeeze(label['shapes'][13]['points'])[0]
                        coords[23] = np.squeeze(label['shapes'][13]['points'])[1]

                    if label['shapes'][14]['label']=='Alt1L': 
                        coords[24] = np.squeeze(label['shapes'][14]['points'])[0]
                        coords[25] = np.squeeze(label['shapes'][14]['points'])[1]

                    if label['shapes'][15]['label']=='Alt1R':
                        coords[26] = np.squeeze(label['shapes'][15]['points'])[0]
                        coords[27] = np.squeeze(label['shapes'][15]['points'])[1]

                    if label['shapes'][16]['label']=='Alt2L': 
                        coords[28] = np.squeeze(label['shapes'][16]['points'])[0]
                        coords[29] = np.squeeze(label['shapes'][16]['points'])[1]

                    if label['shapes'][17]['label']=='Alt2R':
                        coords[30] = np.squeeze(label['shapes'][17]['points'])[0]
                        coords[31] = np.squeeze(label['shapes'][17]['points'])[1]

                    if label['shapes'][18]['label']=='Prn': 
                        coords[32] = np.squeeze(label['shapes'][18]['points'])[0]
                        coords[33] = np.squeeze(label['shapes'][18]['points'])[1]

                    if label['shapes'][19]['label']=='Sn':
                        coords[34] = np.squeeze(label['shapes'][19]['points'])[0]
                        coords[35] = np.squeeze(label['shapes'][19]['points'])[1]
                                                                
                matrix=np.ones(36, dtype=int)
                """
                for i1 in range(0,36):
                    matrix[i1]=450  
                print(np.divide(coords, matrix))
                """                   
                try: 
                    for x in range(25):
                       
                        ########################### Augment Original Images
                        keypoints = [(coords[:2]), (coords[2:4]),(coords[4:6]), (coords[6:8]),
                                    (coords[8:10]), (coords[10:12]), (coords[12:14]), (coords[14:16]), (coords[16:18]), 
                                    (coords[18:20]), (coords[20:22]), (coords[22:24]), (coords[24:26]), (coords[26:28]),
                                    (coords[28:30]), (coords[30:32]), (coords[32:34]), (coords[34:36])]
                        augmented = augmentor(image=img, keypoints=keypoints, class_labels=['NtL','NtR','NbL','NbR',
                                                                                            'NlL','NlR','NmL','NmR','AlL','AlR',
                                                                                            'SbalL','SbalR','Alt1L','Alt1R',
                                                                                            'Alt2L','Alt2R','Prn','Sn'])
                        cv2.imwrite(os.path.join(DS_PATH,'AugDS', partition, 'Real','images', f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])
                        
                        annotation = {}
                        annotation['image'] = image
                        aug_img=augmented['image']
                        annotation['keypoints'] = np.zeros(36,dtype=float)
                        aug_coords=np.zeros(36,dtype=float)  
                        crop_coords=np.zeros(36,dtype=float)
                        matrix=np.ones(36, dtype=int)
                                                
                        if len(augmented['keypoints']) > 0: 
                            for idx, cl in enumerate(augmented['class_labels']):
                                if cl == 'NtL': 
                                    annotation['keypoints'][0] = augmented['keypoints'][idx][0]
                                    annotation['keypoints'][1] = augmented['keypoints'][idx][1]
                                    aug_coords[0]=augmented['keypoints'][idx][0]
                                    aug_coords[1]=augmented['keypoints'][idx][1]
                                if cl == 'NtR': 
                                    annotation['keypoints'][2] = augmented['keypoints'][idx][0]
                                    annotation['keypoints'][3] = augmented['keypoints'][idx][1]
                                    aug_coords[2]=augmented['keypoints'][idx][0]
                                    aug_coords[3]=augmented['keypoints'][idx][1]
                                if cl == 'NbL':
                                    annotation['keypoints'][4] = augmented['keypoints'][idx][0]
                                    annotation['keypoints'][5] = augmented['keypoints'][idx][1]
                                    aug_coords[4]=augmented['keypoints'][idx][0]
                                    aug_coords[5]=augmented['keypoints'][idx][1]
                                if cl == 'NbR':  
                                    annotation['keypoints'][6] = augmented['keypoints'][idx][0]
                                    annotation['keypoints'][7] = augmented['keypoints'][idx][1]
                                    aug_coords[6]=augmented['keypoints'][idx][0]
                                    aug_coords[7]=augmented['keypoints'][idx][1]
                                if cl == 'NlL': 
                                    annotation['keypoints'][8] = augmented['keypoints'][idx][0]
                                    annotation['keypoints'][9] = augmented['keypoints'][idx][1]
                                    aug_coords[8]=augmented['keypoints'][idx][0]
                                    aug_coords[9]=augmented['keypoints'][idx][1]
                                if cl == 'NlR': 
                                    annotation['keypoints'][10] = augmented['keypoints'][idx][0]
                                    annotation['keypoints'][11] = augmented['keypoints'][idx][1]
                                    aug_coords[10]=augmented['keypoints'][idx][0]
                                    aug_coords[11]=augmented['keypoints'][idx][1]
                                if cl == 'NmL': 
                                    annotation['keypoints'][12] = augmented['keypoints'][idx][0]
                                    annotation['keypoints'][13] = augmented['keypoints'][idx][1]
                                    aug_coords[12]=augmented['keypoints'][idx][0]
                                    aug_coords[13]=augmented['keypoints'][idx][1]
                                if cl == 'NmR': 
                                    annotation['keypoints'][14] = augmented['keypoints'][idx][0]
                                    annotation['keypoints'][15] = augmented['keypoints'][idx][1]
                                    aug_coords[14]=augmented['keypoints'][idx][0]
                                    aug_coords[15]=augmented['keypoints'][idx][1]
                                if cl == 'AlL': 
                                    annotation['keypoints'][16] = augmented['keypoints'][idx][0]
                                    annotation['keypoints'][17] = augmented['keypoints'][idx][1]
                                    aug_coords[16]=augmented['keypoints'][idx][0]
                                    aug_coords[17]=augmented['keypoints'][idx][1]
                                if cl == 'AlR':
                                    annotation['keypoints'][18] = augmented['keypoints'][idx][0]
                                    annotation['keypoints'][19] = augmented['keypoints'][idx][1]
                                    aug_coords[18]=augmented['keypoints'][idx][0]
                                    aug_coords[19]=augmented['keypoints'][idx][1]
                                if cl == 'SbalL':
                                    annotation['keypoints'][20] = augmented['keypoints'][idx][0]
                                    annotation['keypoints'][21] = augmented['keypoints'][idx][1]
                                    aug_coords[20]=augmented['keypoints'][idx][0]
                                    aug_coords[21]=augmented['keypoints'][idx][1]
                                if cl == 'SbalR':  
                                    annotation['keypoints'][22] = augmented['keypoints'][idx][0]
                                    annotation['keypoints'][23] = augmented['keypoints'][idx][1]
                                    aug_coords[22]=augmented['keypoints'][idx][0]
                                    aug_coords[23]=augmented['keypoints'][idx][1]
                                if cl == 'Alt1L': 
                                    annotation['keypoints'][24] = augmented['keypoints'][idx][0]
                                    annotation['keypoints'][25] = augmented['keypoints'][idx][1]
                                    aug_coords[24]=augmented['keypoints'][idx][0]
                                    aug_coords[25]=augmented['keypoints'][idx][1]
                                if cl == 'Alt1R': 
                                    annotation['keypoints'][26] = augmented['keypoints'][idx][0]
                                    annotation['keypoints'][27] = augmented['keypoints'][idx][1]
                                    aug_coords[26]=augmented['keypoints'][idx][0]
                                    aug_coords[27]=augmented['keypoints'][idx][1]
                                if cl == 'Alt2L':
                                    annotation['keypoints'][28] = augmented['keypoints'][idx][0]
                                    annotation['keypoints'][29] = augmented['keypoints'][idx][1]
                                    aug_coords[28]=augmented['keypoints'][idx][0]
                                    aug_coords[29]=augmented['keypoints'][idx][1]
                                if cl == 'Alt2R':
                                    annotation['keypoints'][30] = augmented['keypoints'][idx][0]
                                    annotation['keypoints'][31] = augmented['keypoints'][idx][1]
                                    aug_coords[30]=augmented['keypoints'][idx][0]
                                    aug_coords[31]=augmented['keypoints'][idx][1]
                                if cl == 'Prn':  
                                    annotation['keypoints'][32] = augmented['keypoints'][idx][0]
                                    annotation['keypoints'][33] = augmented['keypoints'][idx][1]
                                    aug_coords[32]=augmented['keypoints'][idx][0]
                                    aug_coords[33]=augmented['keypoints'][idx][1]
                                if cl == 'Sn': 
                                    annotation['keypoints'][34] = augmented['keypoints'][idx][0]
                                    annotation['keypoints'][35] = augmented['keypoints'][idx][1]
                                    aug_coords[34]=augmented['keypoints'][idx][0]
                                    aug_coords[35]=augmented['keypoints'][idx][1]
                                
                        matrix=np.ones(36, dtype=int)
                        for i1 in range(0,36):
                            matrix[i1]=450                       
                        annotation['keypoints'] = list(np.divide(annotation['keypoints'], matrix))

                        with open(os.path.join(DS_PATH,'AugDS', partition, 'Real','labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                            json.dump(annotation, f)
                        
                        file_path=  os.path.join(DS_PATH,'AugDS', partition, 'Shift','labels', f'{image.split(".")[0]}.{x}.json')                 
                        shifted_landmarks(aug_coords,450,450,file_path,annotation['image'])     
                        
                        file_path_= os.path.join(DS_PATH,'AugDS', partition, 'Noise','images', f'{image.split(".")[0]}.{x}.jpg')
                        add_noise(augmented['image'],file_path_)    
                        ############################## Resize Original Image
                                                        
                        keypoints1 = [(aug_coords[:2]), (aug_coords[2:4]), (aug_coords[4:6]), (aug_coords[6:8]),(aug_coords[8:10]), 
                                    (aug_coords[10:12]),(aug_coords[12:14]), (aug_coords[14:16]), (aug_coords[16:18]), (aug_coords[18:20]),
                                    (aug_coords[20:22]),(aug_coords[22:24]), (aug_coords[24:26]), (aug_coords[26:28]), (aug_coords[28:30]),
                                    (aug_coords[30:32]),(aug_coords[32:34]), (aug_coords[34:36])]
                            
                        augmented1 = augmentor1(image=aug_img, keypoints=keypoints1, class_labels=['NtL','NtR','NbL','NbR','NlL','NlR','NmL','NmR','AlL','AlR',
                                                                                                    'SbalL','SbalR','Alt1L','Alt1R','Alt2L','Alt2R','Prn','Sn'])

                        
                        cv2.imwrite(os.path.join(DS_PATH,'AugDS', partition,'Real', 'images_resize', f'{image.split(".")[0]}.{x}.jpg'), augmented1['image'])                   
                        #image1=f'{image.split(".")[0]}.jpg'
                        annotation1 = {}
                        annotation1['image'] = image
                        annotation1['keypoints'] = np.zeros(36,dtype=float)
                        resize_coords=np.zeros(36,dtype=float)  
                        matrix=np.ones(36, dtype=int)
            
                        if len(augmented1['keypoints']) > 0: 
                            for idx, cl in enumerate(augmented1['class_labels']):
                                if cl == 'NtL': 
                                    annotation1['keypoints'][0] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][1] = augmented1['keypoints'][idx][1]
                                    resize_coords[0]=augmented1['keypoints'][idx][0]
                                    resize_coords[1]=augmented1['keypoints'][idx][1]
                                if cl == 'NtR': 
                                    annotation1['keypoints'][2] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][3] = augmented1['keypoints'][idx][1]
                                    resize_coords[2]=augmented1['keypoints'][idx][0]
                                    resize_coords[3]=augmented1['keypoints'][idx][1]
                                if cl == 'NbL':
                                    annotation1['keypoints'][4] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][5] = augmented1['keypoints'][idx][1]
                                    resize_coords[4]=augmented1['keypoints'][idx][0]
                                    resize_coords[5]=augmented1['keypoints'][idx][1]
                                if cl == 'NbR':  
                                    annotation1['keypoints'][6] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][7] = augmented1['keypoints'][idx][1]
                                    resize_coords[6]=augmented1['keypoints'][idx][0]
                                    resize_coords[7]=augmented1['keypoints'][idx][1]
                                if cl == 'NlL': 
                                    annotation1['keypoints'][8] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][9] = augmented1['keypoints'][idx][1]
                                    resize_coords[8]=augmented1['keypoints'][idx][0]
                                    resize_coords[9]=augmented1['keypoints'][idx][1]
                                if cl == 'NlR': 
                                    annotation1['keypoints'][10] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][11] = augmented1['keypoints'][idx][1]
                                    resize_coords[10]=augmented1['keypoints'][idx][0]
                                    resize_coords[11]=augmented1['keypoints'][idx][1]
                                if cl == 'NmL': 
                                    annotation1['keypoints'][12] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][13] = augmented1['keypoints'][idx][1]
                                    resize_coords[12]=augmented1['keypoints'][idx][0]
                                    resize_coords[13]=augmented1['keypoints'][idx][1]
                                if cl == 'NmR': 
                                    annotation1['keypoints'][14] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][15] = augmented1['keypoints'][idx][1]
                                    resize_coords[14]=augmented1['keypoints'][idx][0]
                                    resize_coords[15]=augmented1['keypoints'][idx][1]
                                if cl == 'AlL': 
                                    annotation1['keypoints'][16] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][17] = augmented1['keypoints'][idx][1]
                                    resize_coords[16]=augmented1['keypoints'][idx][0]
                                    resize_coords[17]=augmented1['keypoints'][idx][1]
                                if cl == 'AlR':
                                    annotation1['keypoints'][18] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][19] = augmented1['keypoints'][idx][1]
                                    resize_coords[18]=augmented1['keypoints'][idx][0]
                                    resize_coords[19]=augmented1['keypoints'][idx][1]
                                if cl == 'SbalL':
                                    annotation1['keypoints'][20] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][21] = augmented1['keypoints'][idx][1]
                                    resize_coords[20]=augmented1['keypoints'][idx][0]
                                    resize_coords[21]=augmented1['keypoints'][idx][1]
                                if cl == 'SbalR':  
                                    annotation1['keypoints'][22] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][23] = augmented1['keypoints'][idx][1]
                                    resize_coords[22]=augmented1['keypoints'][idx][0]
                                    resize_coords[23]=augmented1['keypoints'][idx][1]
                                if cl == 'Alt1L': 
                                    annotation1['keypoints'][24] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][25] = augmented1['keypoints'][idx][1]
                                    resize_coords[24]=augmented1['keypoints'][idx][0]
                                    resize_coords[25]=augmented1['keypoints'][idx][1]
                                if cl == 'Alt1R': 
                                    annotation1['keypoints'][26] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][27] = augmented1['keypoints'][idx][1]
                                    resize_coords[26]=augmented1['keypoints'][idx][0]
                                    resize_coords[27]=augmented1['keypoints'][idx][1]
                                if cl == 'Alt2L':
                                    annotation1['keypoints'][28] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][29] = augmented1['keypoints'][idx][1]
                                    resize_coords[28]=augmented1['keypoints'][idx][0]
                                    resize_coords[29]=augmented1['keypoints'][idx][1]
                                if cl == 'Alt2R':
                                    annotation1['keypoints'][30] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][31] = augmented1['keypoints'][idx][1]
                                    resize_coords[30]=augmented1['keypoints'][idx][0]
                                    resize_coords[31]=augmented1['keypoints'][idx][1]
                                if cl == 'Prn':  
                                    annotation1['keypoints'][32] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][33] = augmented1['keypoints'][idx][1]
                                    resize_coords[32]=augmented1['keypoints'][idx][0]
                                    resize_coords[33]=augmented1['keypoints'][idx][1]
                                if cl == 'Sn': 
                                    annotation1['keypoints'][34] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][35] = augmented1['keypoints'][idx][1]
                                    resize_coords[34]=augmented1['keypoints'][idx][0]
                                    resize_coords[35]=augmented1['keypoints'][idx][1]
                                
                        
                        for i1 in range(0,36):
                            matrix[i1]=250
                        
                        annotation1['keypoints'] = list(np.divide(annotation1['keypoints'], matrix))
                        
                        with open(os.path.join(DS_PATH,'AugDS', partition, 'Real','labels_resize', f'{image.split(".")[0]}.{x}.json'), 'w') as f1: 
                            json.dump(annotation1, f1)
                            
                        file_path=  os.path.join(DS_PATH,'AugDS', partition, 'Shift','labels_resize', f'{image.split(".")[0]}.{x}.json')                 
                        shifted_landmarks(resize_coords,250,250,file_path,annotation1['image'])     
                        
                        file_path_= os.path.join(DS_PATH,'AugDS', partition, 'Noise','images_resize', f'{image.split(".")[0]}.{x}.jpg')
                        add_noise(augmented1['image'],file_path_)       
                        resize_coords=np.zeros(36,dtype=float)
                        ############################# Crop Augmented Images
                        x_coords=[]
                        y_coords=[]
                        for f in range(36):
                            if f%2==0: x_coords.append(round(aug_coords[f],4))
                            else: y_coords.append(round(aug_coords[f],4))

                        x_min=min(x_coords)
                        x_max=max(x_coords)
                        y_min=min(y_coords)
                        y_max=max(y_coords)
                        
                        y1_crop=int(y_min-9)
                        if y1_crop<0: y1_crop=0
                        y2_crop=int(y_max+15)
                        x1_crop=int(x_min-9)
                        if x1_crop<0: x1_crop=0
                        x2_crop=int(x_max+9)
                        w1=x2_crop-x1_crop
                        h1=y2_crop-y1_crop
                        crop_img=aug_img[y1_crop:y2_crop, x1_crop:x2_crop]           
                        cv2.imwrite(os.path.join(DS_PATH,'AugDS', partition,'Real', 'images_crop', f'{image.split(".")[0]}.{x}.jpg'), crop_img)
                        
                        for i_ in range (36):
                            if i_%2==0: crop_coords[i_]=(aug_coords[i_]-x1_crop)#/w1
                            else: crop_coords[i_]=(aug_coords[i_]-y1_crop)#/h1
                        
                        annotation2 = {}
                        annotation2['image'] = image
                        annotation2['keypoints'] = np.zeros(36,dtype=float)
                        annotation2['keypoints'][0] = crop_coords[0]/w1
                        annotation2['keypoints'][1] = crop_coords[1]/h1
                        annotation2['keypoints'][2] = crop_coords[2]/w1
                        annotation2['keypoints'][3] = crop_coords[3]/h1
                        annotation2['keypoints'][4] = crop_coords[4]/w1
                        annotation2['keypoints'][5] = crop_coords[5]/h1
                        annotation2['keypoints'][6] = crop_coords[6]/w1
                        annotation2['keypoints'][7] = crop_coords[7]/h1
                        annotation2['keypoints'][8] = crop_coords[8]/w1
                        annotation2['keypoints'][9] = crop_coords[9]/h1
                        annotation2['keypoints'][10] = crop_coords[10]/w1
                        annotation2['keypoints'][11] = crop_coords[11]/h1
                        annotation2['keypoints'][12] = crop_coords[12]/w1
                        annotation2['keypoints'][13] = crop_coords[13]/h1
                        annotation2['keypoints'][14] = crop_coords[14]/w1
                        annotation2['keypoints'][15] = crop_coords[15]/h1
                        annotation2['keypoints'][16] = crop_coords[16]/w1
                        annotation2['keypoints'][17] = crop_coords[17]/h1
                        annotation2['keypoints'][18] = crop_coords[18]/w1
                        annotation2['keypoints'][19] = crop_coords[19]/h1
                        annotation2['keypoints'][20] = crop_coords[20]/w1
                        annotation2['keypoints'][21] = crop_coords[21]/h1
                        annotation2['keypoints'][22] = crop_coords[22]/w1
                        annotation2['keypoints'][23] = crop_coords[23]/h1
                        annotation2['keypoints'][24] = crop_coords[24]/w1
                        annotation2['keypoints'][25] = crop_coords[25]/h1
                        annotation2['keypoints'][26] = crop_coords[26]/w1
                        annotation2['keypoints'][27] = crop_coords[27]/h1
                        annotation2['keypoints'][28] = crop_coords[28]/w1
                        annotation2['keypoints'][29] = crop_coords[29]/h1
                        annotation2['keypoints'][30] = crop_coords[30]/w1
                        annotation2['keypoints'][31] = crop_coords[31]/h1
                        annotation2['keypoints'][32] = crop_coords[32]/w1
                        annotation2['keypoints'][33] = crop_coords[33]/h1
                        annotation2['keypoints'][34] = crop_coords[34]/w1
                        annotation2['keypoints'][35] = crop_coords[35]/h1             
                        
                        matrix=np.ones(36, dtype=int)            
                        annotation['keypoints'] = list(np.divide(annotation2['keypoints'], matrix))
                        with open(os.path.join(DS_PATH,'AugDS', partition,'Real', 'labels_crop', f'{image.split(".")[0]}.{x}.json'), 'w') as f: 
                            json.dump(annotation, f)
                            
                        file_path=  os.path.join(DS_PATH,'AugDS', partition, 'Shift','labels_crop', f'{image.split(".")[0]}.{x}.json')                 
                        shifted_landmarks(crop_coords,w1,h1,file_path,annotation2['image'])     
                        
                        file_path_= os.path.join(DS_PATH,'AugDS', partition, 'Noise','images_crop', f'{image.split(".")[0]}.{x}.jpg')
                        add_noise(crop_img,file_path_)
                        
                        ####################### Resized Cropped Images

                        keypoints1 = [(crop_coords[:2]), (crop_coords[2:4]), (crop_coords[4:6]), (crop_coords[6:8]),(crop_coords[8:10]), 
                            (crop_coords[10:12]),(crop_coords[12:14]), (crop_coords[14:16]), (crop_coords[16:18]), (crop_coords[18:20]),
                            (crop_coords[20:22]),(crop_coords[22:24]), (crop_coords[24:26]), (crop_coords[26:28]), (crop_coords[28:30]),
                            (crop_coords[30:32]),(crop_coords[32:34]), (crop_coords[34:36])]
                    
                        augmented1 = augmentor1(image=crop_img, keypoints=keypoints1, class_labels=['NtL','NtR','NbL','NbR','NlL','NlR','NmL','NmR','AlL','AlR',
                                                                                                    'SbalL','SbalR','Alt1L','Alt1R','Alt2L','Alt2R','Prn','Sn'])

                        cv2.imwrite(os.path.join(DS_PATH,'AugDS', partition,'Real', 'images_crop_resize', f'{image.split(".")[0]}.{x}.jpg'), augmented1['image'])
                        
                        #image1=f'{image.split(".")[0]}.jpg'
                        annotation1 = {}
                        annotation1['image'] = image
                        annotation1['keypoints'] = np.zeros(36,dtype=float)
                        resize_coords=np.zeros(36,dtype=float)  
                        
                        
                        if len(augmented1['keypoints']) > 0: 
                            for idx, cl in enumerate(augmented1['class_labels']):
                                if cl == 'NtL': 
                                    annotation1['keypoints'][0] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][1] = augmented1['keypoints'][idx][1]
                                    resize_coords[0]=augmented1['keypoints'][idx][0]
                                    resize_coords[1]=augmented1['keypoints'][idx][1]
                                if cl == 'NtR': 
                                    annotation1['keypoints'][2] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][3] = augmented1['keypoints'][idx][1]
                                    resize_coords[2]=augmented1['keypoints'][idx][0]
                                    resize_coords[3]=augmented1['keypoints'][idx][1]
                                if cl == 'NbL':
                                    annotation1['keypoints'][4] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][5] = augmented1['keypoints'][idx][1]
                                    resize_coords[4]=augmented1['keypoints'][idx][0]
                                    resize_coords[5]=augmented1['keypoints'][idx][1]
                                if cl == 'NbR':  
                                    annotation1['keypoints'][6] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][7] = augmented1['keypoints'][idx][1]
                                    resize_coords[6]=augmented1['keypoints'][idx][0]
                                    resize_coords[7]=augmented1['keypoints'][idx][1]
                                if cl == 'NlL': 
                                    annotation1['keypoints'][8] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][9] = augmented1['keypoints'][idx][1]
                                    resize_coords[8]=augmented1['keypoints'][idx][0]
                                    resize_coords[9]=augmented1['keypoints'][idx][1]
                                if cl == 'NlR': 
                                    annotation1['keypoints'][10] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][11] = augmented1['keypoints'][idx][1]
                                    resize_coords[10]=augmented1['keypoints'][idx][0]
                                    resize_coords[11]=augmented1['keypoints'][idx][1]
                                if cl == 'NmL': 
                                    annotation1['keypoints'][12] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][13] = augmented1['keypoints'][idx][1]
                                    resize_coords[12]=augmented1['keypoints'][idx][0]
                                    resize_coords[13]=augmented1['keypoints'][idx][1]
                                if cl == 'NmR': 
                                    annotation1['keypoints'][14] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][15] = augmented1['keypoints'][idx][1]
                                    resize_coords[14]=augmented1['keypoints'][idx][0]
                                    resize_coords[15]=augmented1['keypoints'][idx][1]
                                if cl == 'AlL': 
                                    annotation1['keypoints'][16] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][17] = augmented1['keypoints'][idx][1]
                                    resize_coords[16]=augmented1['keypoints'][idx][0]
                                    resize_coords[17]=augmented1['keypoints'][idx][1]
                                if cl == 'AlR':
                                    annotation1['keypoints'][18] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][19] = augmented1['keypoints'][idx][1]
                                    resize_coords[18]=augmented1['keypoints'][idx][0]
                                    resize_coords[19]=augmented1['keypoints'][idx][1]
                                if cl == 'SbalL':
                                    annotation1['keypoints'][20] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][21] = augmented1['keypoints'][idx][1]
                                    resize_coords[20]=augmented1['keypoints'][idx][0]
                                    resize_coords[21]=augmented1['keypoints'][idx][1]
                                if cl == 'SbalR':  
                                    annotation1['keypoints'][22] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][23] = augmented1['keypoints'][idx][1]
                                    resize_coords[22]=augmented1['keypoints'][idx][0]
                                    resize_coords[23]=augmented1['keypoints'][idx][1]
                                if cl == 'Alt1L': 
                                    annotation1['keypoints'][24] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][25] = augmented1['keypoints'][idx][1]
                                    resize_coords[24]=augmented1['keypoints'][idx][0]
                                    resize_coords[25]=augmented1['keypoints'][idx][1]
                                if cl == 'Alt1R': 
                                    annotation1['keypoints'][26] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][27] = augmented1['keypoints'][idx][1]
                                    resize_coords[26]=augmented1['keypoints'][idx][0]
                                    resize_coords[27]=augmented1['keypoints'][idx][1]
                                if cl == 'Alt2L':
                                    annotation1['keypoints'][28] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][29] = augmented1['keypoints'][idx][1]
                                    resize_coords[28]=augmented1['keypoints'][idx][0]
                                    resize_coords[29]=augmented1['keypoints'][idx][1]
                                if cl == 'Alt2R':
                                    annotation1['keypoints'][30] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][31] = augmented1['keypoints'][idx][1]
                                    resize_coords[30]=augmented1['keypoints'][idx][0]
                                    resize_coords[31]=augmented1['keypoints'][idx][1]
                                if cl == 'Prn':  
                                    annotation1['keypoints'][32] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][33] = augmented1['keypoints'][idx][1]
                                    resize_coords[32]=augmented1['keypoints'][idx][0]
                                    resize_coords[33]=augmented1['keypoints'][idx][1]
                                if cl == 'Sn': 
                                    annotation1['keypoints'][34] = augmented1['keypoints'][idx][0]
                                    annotation1['keypoints'][35] = augmented1['keypoints'][idx][1]
                                    resize_coords[34]=augmented1['keypoints'][idx][0]
                                    resize_coords[35]=augmented1['keypoints'][idx][1]
                        
                        for i1 in range(0,36):
                            matrix[i1]=250
                        
                        annotation1['keypoints'] = list(np.divide(annotation1['keypoints'], matrix))
                
                        with open(os.path.join(DS_PATH,'AugDS', partition, 'Real','labels_crop_resize', f'{image.split(".")[0]}.{x}.json'), 'w') as f1: 
                            json.dump(annotation1, f1)
                        
                        file_path=  os.path.join(DS_PATH,'AugDS', partition, 'Shift','labels_crop_resize', f'{image.split(".")[0]}.{x}.json')                 
                        shifted_landmarks(resize_coords,250,250,file_path,annotation1['image'])     
                        
                        file_path_= os.path.join(DS_PATH,'AugDS', partition, 'Noise','images_crop_resize', f'{image.split(".")[0]}.{x}.jpg')
                        add_noise(augmented1['image'],file_path_)
                               
                except Exception as e:
                    print(e)
                
                print(rows,": partition: ", partition, f'{image.split(".")[0]}.jpg') 
                rows=rows+1

Prepare_Original_DS()
Prepare_Augmented_DS()

