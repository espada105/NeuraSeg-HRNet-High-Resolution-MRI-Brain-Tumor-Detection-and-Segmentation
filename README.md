# HRNet-Based High-Performance Brain Tumor MRI Classification
HRNet-W48-C-SSLD:	99.69%

This repository contains the implementation code for research that improved brain tumor MRI image classification performance using High-Resolution Network (HRNet).

<table>
  <tr align="center">
    <td width="150px">
      <a href="https://github.com/isshoman123" target="_blank">
        <img src="https://avatars.githubusercontent.com/isshoman123" alt="isshoman123" />
      </a>
    </td>
    <td width="150px">
      <a href="https://github.com/dongsinwoo" target="_blank">
        <img src="https://avatars.githubusercontent.com/dongsinwoo" alt="dongsinwoo" />
      </a>
    </td>
    <td width="150px">
      <a href="https://github.com/espada105" target="_blank">
        <img src="https://avatars.githubusercontent.com/espada105" alt="espada105" />
      </a>
    </td>
  </tr>
  <tr align="center">
    <td>
      Jaewon Kim
    </td>
    <td>
      Dongwoo Shin
    </td>
      <td>
      Seongin Hong
    </td>
  </tr>
</table>

## Research Overview

This study applied HRNet to address the high-resolution information loss problems in existing CNN models (ResNet, VGG, etc.) caused by downsampling in brain tumor MRI image classification. HRNet's multi-resolution parallel processing and feature fusion techniques effectively extracted tumor features of various sizes and shapes, improving classification accuracy.

## Dataset

We used the "Brain Tumor MRI Dataset" from Kaggle, which consists of 7,023 MRI images:
- Glioma: 1,621 images
- Meningioma: 1,645 images
- Pituitary tumor: 1,757 images
- No tumor: 2,000 images

The dataset is divided into Training (5,618 images, about 80%) and Testing (1,405 images, about 20%) folders.

## Training Set Distribution
![image](https://github.com/user-attachments/assets/b813c550-f27c-4274-9100-ca67f996d36f)

## Testing Set Distribution
![image](https://github.com/user-attachments/assets/ee79aeb0-9262-4aca-a94f-90694c20e13d)

## Model Architecture

This study used three HRNet model structures:
- HRNet-W18-C-SSLD
- HRNet-W48-C-SSLD
- HRNet-W64-C

HRNet maintains high-resolution representations throughout the network, processes feature maps of different resolutions in parallel, and exchanges information through fusion modules.

## Implementation Method

### Preprocessing
- Resizing all MRI images to 224×224 pixels
- Normalizing pixel values to [0, 1] range
- Applying data augmentation techniques (RandomHorizontalFlip, RandomAffine, ColorJitter, RandomRotation)

### Model Training
- Optimizer: Adam (learning rate: 0.0007)
- Loss function: Cross Entropy Loss
- Training epochs: 60 (with Early stopping)
- Batch size: 32

## Performance Results

| Model | Precision | Recall | F1-Score | Accuracy(%) |
|-------|-----------|--------|----------|-------------|
| HRNet-W48-C-SSLD | 0.99 | 0.99 | 0.99 | 99.69% |
| HRNet-W18-C-SSLD | 0.99 | 0.99 | 0.99 | 98.93% |
| Keras | 0.98 | 0.98 | 0.98 | 97.87% |
| UNet + VGG11 | 0.97 | 0.97 | 0.97 | 97.84% |
| HRNet-W64-C | 0.97 | 0.97 | 0.97 | 97.71% |
| VGG19 + LSTM + SVM | 0.97 | 0.97 | 0.97 | 97.02% |
| ResNet50 | 0.96 | 0.96 | 0.96 | 96.37% |

- The HRNet-W48-C-SSLD model showed the best performance with 99.69% accuracy.
- It achieved approximately 3.32 percentage points improvement in accuracy compared to ResNet50.

## HRNet-W48-C-ssld Loss Accuracy (Valid)
![image](https://github.com/user-attachments/assets/c5dbe934-2719-4409-8832-a6b2aa284af5)

## Confusion Matrix
![image](https://github.com/user-attachments/assets/7ba4ebc7-5709-40ed-afc6-50490e7fbc4a)

## Conclusion

Through this research, we confirmed that the HRNet structure provides superior performance compared to existing CNN models in brain tumor MRI image classification and has excellent ability to detect small tumors, which is important for early diagnosis. This suggests that high-resolution representation learning has great potential in medical image analysis, particularly in important medical areas such as brain tumor diagnosis.

<hr>

# HRNet 기반 고성능 뇌종양 MRI 영상 분류

본 저장소는 고해상도 네트워크(HRNet)를 활용하여 뇌종양 MRI 영상 분류 성능을 향상시킨 연구 결과를 구현한 코드를 포함하고 있습니다.

<table>
  <tr align="center">
    <td width="150px">
      <a href="https://github.com/isshoman123" target="_blank">
        <img src="https://avatars.githubusercontent.com/isshoman123" alt="isshoman123" />
      </a>
    </td>
    <td width="150px">
      <a href="https://github.com/dongsinwoo" target="_blank">
        <img src="https://avatars.githubusercontent.com/dongsinwoo" alt="dongsinwoo" />
      </a>
    </td>
    <td width="150px">
      <a href="https://github.com/espada105" target="_blank">
        <img src="https://avatars.githubusercontent.com/espada105" alt="espada105" />
      </a>
    </td>
  </tr>
  <tr align="center">
    <td>
      김재원
    </td>
    <td>
      신동우
    </td>
      <td>
      홍성인
    </td>
  </tr>
</table>

## 연구 개요

본 연구는 뇌종양 MRI 영상 분류에 있어 기존 CNN 모델(ResNet, VGG 등)의 다운샘플링으로 인한 고해상도 정보 손실 문제를 해결하기 위해 HRNet을 적용했습니다. HRNet의 다중 해상도 병렬 처리와 특징 융합 기법은 다양한 크기와 형태의 뇌종양 특징을 효과적으로 추출하여 분류 정확도를 향상시켰습니다.
HRNet-W48-C-SSLD:	99.69%

## 데이터셋

Kaggle의 "Brain Tumor MRI Dataset"을 사용했으며, 총 7,023개의 MRI 영상으로 구성되어 있습니다:
- 교모종(glioma): 1,621개
- 수막종(meningioma): 1,645개
- 뇌하수체 종양(pituitary): 1,757개
- 정상(no tumor): 2,000개

데이터셋은 Training(5,618개, 약 80%)과 Testing(1,405개, 약 20%) 폴더로 분리되어 있습니다.

## Training Set Distribution
![image](https://github.com/user-attachments/assets/b813c550-f27c-4274-9100-ca67f996d36f)

## Testing Set Distribution
![image](https://github.com/user-attachments/assets/ee79aeb0-9262-4aca-a94f-90694c20e13d)

## 모델 구성

본 연구에서는 세 가지 HRNet 모델 구조를 사용했습니다:
- HRNet-W18-C-SSLD
- HRNet-W48-C-SSLD
- HRNet-W64-C

HRNet은 네트워크 전체에 걸쳐 고해상도 표현을 유지하며, 다양한 해상도의 특징 맵을 병렬로 처리하고 융합 모듈을 통해 정보를 교환하는 구조를 가지고 있습니다.

## 구현 방법

### 전처리 과정
- 모든 MRI 영상을 224×224 픽셀로 리사이징
- 픽셀 값을 [0, 1] 범위로 정규화
- 데이터 증강 기법 적용 (RandomHorizontalFlip, RandomAffine, ColorJitter, RandomRotation)

### 모델 학습
- 옵티마이저: Adam (learning rate: 0.0007)
- 손실 함수: Cross Entropy Loss
- 학습 에폭: 60 (Early stopping 적용)
- 배치 사이즈: 32

## 성능 결과

| Model | Precision | Recall | F1-Score | Accuracy(%) |
|-------|-----------|--------|----------|-------------|
| HRNet-W48-C-SSLD | 0.99 | 0.99 | 0.99 | 99.69% |
| HRNet-W18-C-SSLD | 0.99 | 0.99 | 0.99 | 98.93% |
| Keras | 0.98 | 0.98 | 0.98 | 97.87% |
| UNet + VGG11 | 0.97 | 0.97 | 0.97 | 97.84% |
| HRNet-W64-C | 0.97 | 0.97 | 0.97 | 97.71% |
| VGG19 + LSTM + SVM | 0.97 | 0.97 | 0.97 | 97.02% |
| ResNet50 | 0.96 | 0.96 | 0.96 | 96.37% |

- HRNet-W48-C-SSLD 모델이 99.69%의 정확도로 가장 우수한 성능을 보였습니다.
- 특히 ResNet50과 비교하여 약 3.32%p의 정확도 향상을 달성했습니다.



## HRNet-W48-C-ssld Loss Accuracy (Valid)
![image](https://github.com/user-attachments/assets/c5dbe934-2719-4409-8832-a6b2aa284af5)

## Confusion Matrix
![image](https://github.com/user-attachments/assets/7ba4ebc7-5709-40ed-afc6-50490e7fbc4a)


## 결론

본 연구를 통해 HRNet 구조가 뇌종양 MRI 영상 분류에 있어 기존 CNN 모델들보다 우수한 성능을 제공하며, 특히 조기 진단에 중요한 작은 크기의 종양 감지에 탁월한 능력을 가지고 있음을 확인했습니다. 이는 고해상도 표현 학습이 의료 영상 분석, 특히 뇌종양 진단과 같은 중요한 의료 영역에서 큰 잠재력을 가지고 있음을 시사합니다.


