# 🔪 Knife Classification Under Environmental Variations Using CNN Models

조명, 배경, 가림 같은 환경 조건이 바뀔 때 CNN 모델이 칼 종류를 얼마나 안정적으로 분류하는지 비교한 프로젝트입니다.

📄 **논문**: *Knife Classification Under Environmental Variations Using CNN Models: A Study on Robustness in Multi-Condition Knife Image Recognition Tasks* (MITA 2025)

> CCTV 영상에서 칼을 놓치면 큰 사고로 이어질 수 있습니다.
> 그래서 평균 정확도가 아니라 **실제 배치될 환경에서 얼마나 버티는지**를 기준으로 모델을 골라야 한다고 봤습니다.

---

## 결론

- 모든 환경에서 가장 좋은 모델은 없었습니다.
- 평균 정확도가 높다고 실제 환경에서도 믿을 만한 건 아니었습니다.
- 모델은 조명, 가림, 기기 성능처럼 **배치할 곳의 조건**을 보고 골라야 합니다.

| 이런 환경이라면 | 추천 모델 | 이유 | 크기 |
|---|---|---|---|
| 조명·배경이 자주 바뀜 | DenseNet121 | 모든 조건에서 가장 고르게 버팀 | ~30MB |
| 커터칼만 잡으면 되는 경보 시스템 | EfficientNetB0 | 커터칼은 정확하게 잡고 가벼움 | ~20MB |
| 어두운 곳, 야간 감시 | ResNet50 | 어두운 장면에서 정확도가 더 높음 | ~100MB |
| 비교 기준, 파일럿 테스트 | VGG16 | 정확도가 고르게 나와서 기준 모델로 쓰기 좋음 | ~500MB |

---

## 개발 과정

### Phase 1. 파일럿 실험

YOLOv8로 칼을 탐지하고 조명을 바꿔 가며 테스트했습니다.

- 직접 찍은 영상 3개에서 뽑은 89프레임
- 장소 한 곳, 칼 한 종류
- 결과: mAP@0.5 = 0.972

점수는 높았지만 장소도 칼도 하나뿐이었습니다. 실제 CCTV 환경에서도 이 점수가 나올지는 알 수 없었습니다.

### Phase 2. 조건별 평가

영상을 19개로 늘리고, 같은 조건에서 CNN 모델 4개를 학습해 환경 조건별로 성능을 나눠 봤습니다.

- 모델: VGG16, ResNet50, DenseNet121, EfficientNetB0
- 조건: 조명, 배경, 가림, 칼 종류

### Phase 3. 결과 분석

**전체 정확도**

| 모델 | Train | Validation |
|---|---|---|
| VGG16 | 94.76% | 96.00% |
| ResNet50 | 57.40% | 64.00% |
| DenseNet121 | **99.14%** | **100.00%** |
| EfficientNetB0 | 36.64% | 36.00% |

<!-- 전체 정확도 막대그래프 -->
<img width="428" height="278" alt="image" src="https://github.com/user-attachments/assets/f0cd2df5-bda6-420b-9f7c-930e857cd4e1" />


**조건별 정확도**

DenseNet121과 EfficientNetB0이 가장 대조적이어서 두 모델을 비교했습니다.

<!-- 조건별 그래프 4개 (Lighting / Occlusion / Background / Knife Type) -->
<img width="914" height="651" alt="image" src="https://github.com/user-attachments/assets/078a9e9d-2bff-4992-9df6-91393cb61bf9" />


| 조건 | DenseNet121 | EfficientNetB0 |
|---|---|---|
| 조명 | 밝을 때와 어두울 때 차이가 거의 없음 | 어두운 장면에서 13.7%까지 떨어짐 |
| 가림 | 일부가 가려져도 안정적 | 가림 여부와 상관없이 계속 낮음 |
| 배경 | 배경이 달라도 고르게 맞힘 | 창가 100%, 주방 4.6%로 배경마다 크게 흔들림 |
| 칼 종류 | 세 종류를 고르게 맞힘 | 커터칼만 맞히고 나머지는 거의 못 맞힘 |

**Confusion Matrix**

DenseNet121이 클래스끼리 가장 덜 헷갈렸고, 특히 식칼과 과도를 잘 구분했습니다. VGG16은 식칼과 과도를 자주 헷갈렸고, ResNet50과 EfficientNetB0은 대부분을 커터칼로 예측했습니다.

**학습 곡선**

DenseNet121과 VGG16은 과적합 없이 안정적으로 수렴했습니다. ResNet50은 학습 정확도부터 낮아 과소적합됐고, EfficientNetB0은 초반부터 학습이 멈췄습니다.

<details>
<summary>Confusion Matrix · 학습 곡선 보기</summary>
<img width="671" height="551" alt="image" src="https://github.com/user-attachments/assets/7acfa94c-3f78-47ff-b5e1-cf571e889266" />
<img width="626" height="506" alt="image" src="https://github.com/user-attachments/assets/909c0166-52cb-4956-a5bd-790cd29e25bb" />



</details>

---

## 데이터셋

### 만드는 방법
- 실내에서 직접 찍은 영상 19개에서 YOLOv8(`yolov8s.pt`)로 칼 부분을 프레임마다 잘라냈습니다.
- 그렇게 모은 이미지가 약 15,800장입니다.
- Train / Validation / Test를 6:2:2로 나눴고, Test는 약 3,160장입니다.
- 촬영 조건은 폴더명에 넣었습니다.

```
knife{번호}_{조명}_{배경}_{칼 종류}_{가림}
예) knife05_dark_kitchen_과도_yes
```

### 클래스
- **식칼**: 큰 주방용 칼
- **과도**: 작은 과일칼
- **커터칼**: 박스 자르는 칼

### 조건별 영상 수

| 조건 | 값 | 영상 수 |
|---|---|---|
| 조명 | bright / dark | 12 / 7 |
| 배경 | kitchen / desk / livingroom / floor / window | 3 / 3 / 3 / 4 / 4 |
| 가림 | yes / no | 10 / 9 |
| 칼 종류 | 식칼 / 과도 / 커터칼 | 5 / 5 / 9 |

### 다운로드
```
https://github.com/jwdebbie/knife-classification/raw/main/knives_dataset.zip
```

---

## 모델 구성

네 모델 모두 같은 방식으로 전이학습했고, 데이터 분할과 설정도 똑같이 맞췄습니다.

```
ImageNet 사전학습 백본 (frozen)
  → GlobalAveragePooling2D
  → Dense(128, relu)
  → Dense(3, softmax)   # 식칼 / 과도 / 커터칼
```

| 항목 | 값 |
|---|---|
| 옵티마이저 | Adam (lr = 1e-4) |
| 손실 함수 | categorical_crossentropy |
| 입력 크기 | 224×224 RGB |
| 배치 크기 | 32 |
| 에폭 | 10 |
| 데이터 분할 | Train / Val / Test = 6:2:2 |
| 전처리 | ImageDataGenerator (전처리 + augmentation) |

---

## 실행 방법 (Google Colab)

### 1. 데이터셋 만들기
Google Drive에 `knife_videos/` 폴더와 영상 파일을 넣은 뒤 `Dataset_preparation.py`를 실행합니다.

### 2. 학습과 평가
모델 스크립트를 Colab에서 실행하면 데이터셋을 GitHub에서 받아 학습하고, 조건별 정확도까지 분석합니다.

각 스크립트가 만드는 결과는 다음과 같습니다.
- 학습/검증 정확도 곡선
- Confusion Matrix
- 조명·배경·가림·칼 종류별 정확도
- 영상별 정확도 (조건 조합 분석)
- `{model}_results.csv`

### 패키지
```
tensorflow
ultralytics
opencv-python
scikit-learn
matplotlib
pandas
```

---

## 폴더 구조

```
Robustness-Research/
├── Dataset_preparation.py   # 영상에서 칼 이미지 자동 추출 (YOLOv8)
├── Vgg16.py                 # VGG16 학습 + 조건별 정확도 분석
├── Resnet50.py              # ResNet50 학습 + 조건별 정확도 분석
├── Densenet121.py           # DenseNet121 학습 + 조건별 정확도 분석
└── Efficientnetb0.py        # EfficientNetB0 학습 + 조건별 정확도 분석
```

---

## 앞으로 할 일

- YOLO 같은 탐지 모델까지 붙여서 실험
- 실시간 환경에서 직접 돌려 보기
- 데이터셋을 늘려 일반화 성능 높이기

---

## 논문

```
J. Lee, D. Lee, S. Oh, S. W. An, B. R. Jeon, Y. H. Park,
"Knife Classification Under Environmental Variations Using CNN Models:
A Study on Robustness in Multi-Condition Knife Image Recognition Tasks,"
MITA 2025.
```
