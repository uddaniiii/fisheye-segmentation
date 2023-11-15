#!/usr/bin/env python
# coding: utf-8

# # Import

# In[1]:


import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models.segmentation as segmentation
import segmentation_models_pytorch as smp
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
from albumentations.core.composition import Compose

from torchvision import transforms
# import imgaug.augmenters as iaa
from PIL import Image
import imageio
from math import sqrt
from albumentations.core.transforms_interface import DualTransform
from albumentations.core.transforms_interface import BasicTransform
# from pytorch_toolbelt import losses as L
from FisheyeSeg_master.data.FishEyeGenerator import FishEyeGenerator
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# # RLE Encoding

# In[3]:


# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# # Custom Dataset

# In[4]:


class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image
        
        mask_path = self.data.iloc[idx, 2]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask == 255] = 12 #배경을 픽셀값 12로 간주

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


# # Fisheye Lens Conversion


# In[15]:


old_param=None
old_focal=None
cnt=0
old_img_path="None"

class Fisheye(DualTransform):
    def __init__(self, csv_path="./train_source.csv", focal_len=350, dst_size=[1024, 2048], 
                 **kwargs):
        super().__init__(**kwargs)
        self.csv_path = csv_path
        data = pd.read_csv(csv_path)
        self.img_paths = data['id'].tolist()
        self.img_path = None  # 초기화 추가

        # self.focal_len=focal_len

        self._generator = FishEyeGenerator(focal_len, dst_size)
        self._F_RANGE = [200, 350]
        self._EXT_PARAM_RANGE = [5, 5, 10, 0.3, 0.3, 0.4]
        self._generator.set_bkg(bkg_label=12,bkg_color=[0,0,0])

    def set_ext_param_range(self, ext_param):
        self._EXT_PARAM_RANGE = ext_param
        self._generator.set_ext_param_range(ext_param)

    def rand_ext_params(self):
        self._generator.rand_ext_params()

    def set_ext_params(self, ext_params):
        self._generator.set_ext_params(ext_params)

    def set_f(self, focal_len):
        # global focal_leng
        # print(type(focal_len))
        self._generator.set_f(focal_len)

    def rand_f(self, f_range=[200, 400]):
        self._F_RANGE = f_range
        self._generator.rand_f(f_range)

    def generate(self, img, flag, img_id, old_img_id):
        global old_param, old_focal, old_img_path
        src_image = img

        if img_id==old_img_id:
            self._F_RAND_FLAG=False
            self._EXT_RAND_FLAG=False
        else:
            self._F_RAND_FLAG=True
            self._EXT_RAND_FLAG=True

        # print("f_land flag 상태:", self._F_RAND_FLAG,"ext rand flag:" ,self._EXT_RAND_FLAG)

        if self._F_RAND_FLAG:
            randf=self._generator.rand_f(self._F_RANGE)
            old_focal=randf
        elif not self._F_RAND_FLAG:
            self._generator.set_f(old_focal)

        if self._EXT_RAND_FLAG:
            randext=self._generator.rand_ext_params()
            # print("ext param true")
            # self._generator.print_ext_param()
            old_param=randext
        elif not self._EXT_RAND_FLAG:
            [self._generator._alpha, self._generator._beta, self._generator._theta, 
             self._generator._x_trans, self._generator._y_trans, self._generator._z_trans]=\
            old_param
            # print("ext param false")
            # self._generator.print_ext_param()

        if flag == 3:
            result = self._generator.transFromColor(src_image)
        elif flag == 0:
            result = self._generator.transFromGray(src_image)
        else:
            raise ValueError("Unsupported image flag")

        return result.astype(np.uint8)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        global cnt, old_img_path
        if cnt < len(self.img_paths):
            self.img_path = self.img_paths[cnt]
        else:
            cnt = 0

        if self.img_path == old_img_path:
            cnt += 1

        if len(img.shape) == 2:
            flag = 0
        elif len(img.shape) == 3 and img.shape[2] == 3:
            flag = 3
        else:
            raise ValueError("Unsupported image shape")

        fe = self.generate(img, flag, self.img_path, old_img_path)
        old_img_path = self.img_path
        return fe

    def get_transform_init_args_names(self):
        return ('csv_path', 'focal_len', 'dst_size')


# # Check Data Augmentation

# In[16]:


# for epoch in tqdm(range(50)): # 에폭

#     # 데이터 증강을 위한 transform 파이프라인 정의
#     transform = A.Compose([
#         # A.ColorJitter(p=0.3),
#         # A.Downscale(p=0.3),
#         # A.Equalize(p=0.3),
#         # A.HueSaturationValue(p=0.3),
#         # A.ToGray(p=0.15),
#         Fisheye(p=1.0),
#         # A.Spatter(p=1.0),
#         # A.Defocus(p=1.0),
#         # # A.Perspective(p=1.0),
#         # # A.CenterCrop(height=672,width=1344,p=1.0),
#         # A.Resize(512, 1024),
#         # A.Normalize(),
#         # ToTensorV2()
#     ])
#     dataset = CustomDataset(csv_file='./train_source.csv') # dataset 불러오기
#     aug_dataset = CustomDataset(csv_file='./train_source.csv', transform=transform) 
# # dataset 불러오기

#     image,mask=dataset.__getitem__(epoch)
#     aug_image,aug_mask=aug_dataset.__getitem__(epoch)

#     # print(aug_image.size())
#     # print(aug_mask.size())

#     plt.figure(figsize=(15,10))
#     plt.subplot(1, 3, 1)
#     plt.imshow(image)
#     plt.title("Original Image")

#     plt.subplot(1, 3, 2)
#     # plt.imshow(aug_image)
#     plt.imshow(aug_image.transpose(0,1,2))  # Transpose to (height, width, channels)
#     plt.title("Augmented Image")

#     plt.subplot(1, 3, 3)
#     # plt.imshow(aug_mask)
#     plt.imshow(aug_mask)  # 이미지의 차원 순서가 (height, width, channels)인 경우
#     plt.title("Augmented Mask")

#     plt.show()


# # Model Define

# In[17]:


model = smp.FPN(encoder_name="mit_b0",  # 필수 파라미터: 사용할 인코더 백본의 이름
    in_channels=3,    
    classes=13,        
    encoder_weights="imagenet"  
)
model = model.to(device)

criterion = smp.losses.FocalLoss("multiclass") 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4) 


# # Save & Load Model

# In[18]:


save_dir = "./save/fpn_mvt_real_aug/"  # 모델 저장 디렉토리
model_name = "trained_epoch{}.pth"  # 모델 파일 이름 패턴

# 훈련된 모델을 저장하는 함수
def save_model(model, epoch):
    save_path = save_dir + model_name.format(epoch)
    torch.save(model.state_dict(), save_path)
    print(f"Epoch {epoch} 모델 저장이 완료되었습니다.")

# 모델 불러오는 함수
def load_model(model, load_path):
    state_dict = torch.load(load_path)
    # 이전에 저장된 모델과 현재 모델 간 레이어 일치 여부 확인
    model_dict = model.state_dict()
    new_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("모델 불러오기가 완료되었습니다.")


# # Check Model Input/Output Size

# In[19]:


# transform = A.Compose(
#     [   
#         # Fisheye(p=0.7),
#         A.Resize(512, 1024),
#         A.Normalize(),  # 이미지 픽셀값 정규화
#         ToTensorV2()
#     ]
# )

# val_transform = A.Compose(
#     [   
#         Fisheye(p=0.7),
#         A.Resize(512, 1024),
#         A.Normalize(),  # 이미지 픽셀값 정규화
#         ToTensorV2()
#     ]
# )

# dataset = CustomDataset(csv_file='./train_source.csv', transform=transform)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# val_dataset = CustomDataset(csv_file='./train_source.csv', transform=val_transform)  
# # validation 데이터셋 로딩
# val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)

# for images, masks in tqdm(val_dataloader):
#     images = images.float().to(device)
#     masks = masks.long().to(device)

#     optimizer.zero_grad()
#     outputs = model(images)

#     print(outputs.size(), masks.size())
#     loss = criterion(outputs, masks)


# # Train Model

# In[20]:


train_losses = []  # 훈련 손실을 저장하는 리스트
val_losses = []  # 검증 손실을 저장하는 리스트
train_accuracies = []  # 훈련 정확도를 저장하는 리스트
val_accuracies = []  # 검증 정확도를 저장하는 리스트


# In[21]:


# load_model(model, "./save/psp_dense_dice_aug/trained_epoch22.pth")

best_val_accuracy = float('-inf')
patience = 20  # 최소 검증 손실값이 patience 번째 이후로 개선되지 않으면 학습 중단
counter = 0

for epoch in range(50):
    epoch+=1
    model.train()
    epoch_loss = 0
    correct_pixels_train = 0
    total_pixels_train = 0

    transform = A.Compose(
        [   
            A.ColorJitter(p=0.3),
            A.Downscale(p=0.3),
            A.Equalize(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.ToGray(p=0.15),
            Fisheye(p=1.0, csv_path="./train_source.csv"),
            A.Spatter(p=0.25),
            A.Defocus(p=0.1),
            A.Perspective(p=0.2),
            A.Resize(256, 512),
            A.Normalize(),
            ToTensorV2()
        ]
    )

    val_transform = A.Compose(
        [   
            A.ColorJitter(p=0.3),
            A.Downscale(p=0.3),
            A.Equalize(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.ToGray(p=0.15),
            Fisheye(p=1.0, csv_path="./val_source.csv"),
            A.Spatter(p=0.25),
            A.Defocus(p=0.1),
            A.Perspective(p=0.2),
            A.Resize(256, 512),
            A.Normalize(),
            ToTensorV2()

        ]
    )

    dataset = CustomDataset(csv_file='./train_source.csv', transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    val_dataset = CustomDataset(csv_file='./val_source.csv', transform=val_transform)  
    # validation 데이터셋 로딩
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)

    for images, masks in tqdm(dataloader):
        images = images.float().to(device)
        masks = masks.long().to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        predicted_masks_train = torch.argmax(outputs, dim=1)  # Get predicted class labels
        correct_pixels_train += (predicted_masks_train == masks).sum().item()
        total_pixels_train += masks.numel()

    epoch_loss /= len(dataloader)
    accuracy_train = correct_pixels_train / total_pixels_train

    # Validation loop
    model.eval()
    total_validation_loss = 0.0
    correct_pixels_val = 0
    total_pixels_val = 0

    with torch.no_grad():
        for val_images, val_masks in val_dataloader:
            val_images = val_images.float().to(device)
            val_masks = val_masks.long().to(device)

            val_outputs = model(val_images)
            val_loss = criterion(val_outputs, val_masks)

            total_validation_loss += val_loss.item() * val_images.size(0)

            predicted_masks_val = torch.argmax(val_outputs, dim=1)  
            correct_pixels_val += (predicted_masks_val == val_masks).sum().item()
            total_pixels_val += val_masks.numel()

    average_validation_loss = total_validation_loss / len(val_dataset)
    accuracy_val = correct_pixels_val / total_pixels_val

    if accuracy_val > best_val_accuracy:
        best_val_accuracy = accuracy_val
        counter = 0
        save_path = save_dir + model_name.format(epoch)
        torch.save(model.state_dict(), save_path)
        print(f"Epoch {epoch} 모델 저장이 완료되었습니다.")
    else:
        counter += 1

    if counter >= patience:
        print(f"검증 정확도가 {patience} 에폭 이상 개선되지 않아 학습을 중단합니다.")
        break

    print(f'Epoch {epoch}, Training Loss: {epoch_loss}, Training Accuracy: {accuracy_train}, \
          Validation Loss: {average_validation_loss}, Validation Accuracy: {accuracy_val}')

    train_losses.append(epoch_loss)
    val_losses.append(average_validation_loss)

    train_accuracies.append(accuracy_train)
    val_accuracies.append(accuracy_val)

    train_losses_np = torch.tensor(train_losses).cpu().numpy()
    val_losses_np = torch.tensor(val_losses).cpu().numpy()
    train_accuracies_np = torch.tensor(train_accuracies).cpu().numpy()
    val_accuracies_np = torch.tensor(val_accuracies).cpu().numpy()

    plt.plot(range(1, len(train_losses_np) + 1), train_losses_np, label='Train Loss')
    plt.plot(range(1, len(val_losses_np) + 1), val_losses_np, label='Validation Loss')
    plt.plot(range(1, len(train_accuracies_np) + 1), train_accuracies_np, 
             label='Train Accuracy')
    plt.plot(range(1, len(val_accuracies_np) + 1), val_accuracies_np, 
             label='Validation Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Loss & Accuracy')
    plt.legend()
    plt.show()


# In[ ]:


# print(train_losses)
# print(val_losses)

# train_losses.pop()
# val_losses.pop()

# print(len(train_losses))
# print(val_losses)


# # Load Model

# In[22]:


# 모델 생성
model = smp.FPN(encoder_name="mit_b0",  # 필수 파라미터: 사용할 인코더 백본의 이름
    in_channels=3,    
    classes=13,        
    encoder_weights="imagenet"  
)

# 저장된 모델의 파라미터 불러오기 (strict=False 옵션 사용)
state_dict = torch.load('./save/fpn_mvt_real_aug/trained_epoch3.pth', 
                        map_location=torch.device('cpu'))

# 저장된 모델의 클래스 수 (1개의 클래스일 때)
saved_num_classes = 13

# 현재 모델의 클래스 수 (예시로 21로 설정, 실제 사용하는 클래스 수로 수정)
current_num_classes = 13

# 모델의 분류기 레이어 크기 변경
if saved_num_classes != current_num_classes:
    # 모델의 분류기 레이어를 1x1 컨볼루션 레이어로 수정
    model.classifier[4] = torch.nn.Conv2d(256, current_num_classes, kernel_size=(1, 1), 
                                          stride=(1, 1))
    # 모델의 분류기 레이어를 초기화
    torch.nn.init.xavier_uniform_(model.classifier[4].weight)  # 또는 다른 초기화 방법 사용

# 모델의 파라미터 로드
model.load_state_dict(state_dict, strict=False)

# GPU 사용이 가능한 경우에는 GPU로 데이터 이동
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# # Inference

# In[23]:


# albumentations 라이브러리를 사용하여 이미지 데이터에 대한 변환(transform) 파이프라인 정의
transform = A.Compose(
    [
        # Fisheye(p=1.0),
        A.Resize(256, 512), # 이미지 크기 조정
        A.Normalize(),  # 이미지 픽셀값 정규화
        ToTensorV2() # 이미지를 텐서로 변환
    ]
)

test_dataset = CustomDataset(csv_file='./test.csv', transform=transform, infer=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

with torch.no_grad():
    model.eval()
    result = []
    for images in tqdm(test_dataloader):
        images = images.float().to(device)
        outputs = model(images)
        outputs = torch.softmax(outputs, dim=1).cpu()
        outputs = torch.argmax(outputs, dim=1).numpy()
        # batch에 존재하는 각 이미지에 대해서 반복
        for idx, pred in enumerate(outputs):
            pred = pred.astype(np.uint8)
            pred = Image.fromarray(pred) # 이미지로 변환
            pred = pred.resize((960, 540), Image.NEAREST) # 960 x 540 사이즈로 변환
            pred = np.array(pred) # 다시 수치로 변환
            
            # 원본 이미지 및 예측 이미지 출력
            plt.figure(figsize=(15, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(images[idx].cpu().numpy().transpose(1, 2, 0))
            plt.title("Original Image")
            
            plt.subplot(1, 2, 2)
            plt.imshow(pred) # 예측 이미지의 cmap을 'jet'으로 설정하여 클래스 구분을 쉽게 함
            plt.title("Predicted Mask")
            
            plt.show()
            
            # class 0 ~ 11에 해당하는 경우에 마스크 형성 / 12(배경)는 제외하고 진행
            for class_id in range(12):
                class_mask = (pred == class_id).astype(np.uint8)
                if np.sum(class_mask) > 0: # 마스크가 존재하는 경우 encode
                    mask_rle = rle_encode(class_mask)
                    result.append(mask_rle)
                else: # 마스크가 존재하지 않는 경우 -1
                    result.append(-1)


# # Visualize Validation

# In[ ]:


# albumentations 라이브러리를 사용하여 이미지 데이터에 대한 변환(transform) 파이프라인 정의
transform = A.Compose(
    [
        Fisheye(p=1.0, csv_path="./val_source.csv"),
        # A.CenterCrop(height=672,width=1344,p=1.0),
        A.Resize(256, 512), # 이미지 크기 조정
        A.Normalize(),  # 이미지 픽셀값 정규화
        ToTensorV2() # 이미지를 텐서로 변환
    ]
)

test_dataset = CustomDataset(csv_file='./val_source.csv', transform=transform, infer=False)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

with torch.no_grad():
    model.eval()
    result = []
    for images, masks in tqdm(test_dataloader):
        images = images.float().to(device)
        img1 = np.transpose(np.array(images[0, :, :, :].to('cpu')), (1, 2, 0))

        outputs = model(images)
        outputs = torch.softmax(outputs, dim=1).cpu()
        outputs = torch.argmax(outputs, dim=1).numpy()

        for pred in outputs:
            pred = pred.astype(np.uint8)
            pred_resized = Image.fromarray(pred).resize((960, 540), Image.NEAREST)
            pred_resized = np.array(pred_resized)

            mask1 = np.array(masks[0, :, :])
            mask1_resized = Image.fromarray(mask1).resize((960, 540), Image.NEAREST)
            mask1_resized = np.array(mask1_resized)

            for class_id in range(12):
                class_mask = (pred_resized == class_id).astype(np.uint8)
                if np.sum(class_mask) > 0:
                    mask_rle = rle_encode(class_mask)
                    result.append(mask_rle)
                else:
                    result.append(-1)
            
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(img1)
        plt.title("Image")

        plt.subplot(1, 3, 2)
        plt.imshow(pred_resized)
        plt.title("Prediction")

        plt.subplot(1, 3, 3)
        plt.imshow(mask1_resized)
        plt.title("Mask")
        
        plt.show()


# # Submission

# In[ ]:


submit = pd.read_csv('./sample_submission.csv')
submit['mask_rle'] = result
submit


# In[ ]:


submit.to_csv('./submit.csv', index=False)