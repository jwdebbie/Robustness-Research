# 🔪 Robust Perception Analysis through knife classification

다양한 환경 조건(조도, 배경, 오클루전)에서 딥러닝 모델이 칼 종류를 얼마나 강인하게 분류하는지 분석한 프로젝트입니다.

---

## 프로젝트 개요

CCTV·보안 카메라 영상처럼 **통제되지 않은 환경**에서 칼을 정확히 탐지·분류하는 것은 실제 보안 시스템에서 중요한 과제입니다. 
본 프로젝트는 아래 4가지 CNN 기반 전이학습 모델의 분류 성능을 **환경 변수별로 비교**합니다.

| 모델 | 파일 |
|---|---|
| VGG16 | `Vgg16.py` |
| ResNet50 | `Resnet50.py` |
| DenseNet121 | `Densenet121.py` |
| EfficientNetB0 | `Efficientnetb0.py` |

---

## 파일 구조

```
01_Robust_Perception_Analysis/
├── Dataset_preparation.py   # 영상에서 칼 이미지 자동 추출 (YOLOv8)
├── Vgg16.py                 # VGG16 전이학습 + 조건별 정확도 분석
├── Resnet50.py              # ResNet50 전이학습 + 조건별 정확도 분석
├── Densenet121.py           # DenseNet121 전이학습 + 조건별 정확도 분석
└── Efficientnetb0.py        # EfficientNetB0 전이학습 + 조건별 정확도 분석
```

---

## 데이터셋

### 수집 방법
- 직접 촬영한 영상 19개에서 YOLOv8(`yolov8s.pt`)을 이용해 칼 영역을 자동 크롭
- 영상 1개당 최대 50장 추출 → 폴더명에 촬영 조건 인코딩

### 폴더명 구조
```
knife{번호}_{조도}_{배경}_{칼종류}_{오클루전}.mp4
예) knife05_dark_kitchen_과도_yes
```

### 분류 클래스 (3종)
- **식칼** — 큰 주방용 칼
- **과도** — 작은 과일칼
- **커터칼** — 커터(박스칼)

### 환경 변수
| 변수 | 값 |
|---|---|
| 조도 | `bright` / `dark` |
| 배경 | `kitchen` / `desk` / `livingroom` / `floor` / `window` |
| 오클루전 | `yes` (손이나 물체로 가려짐) / `none` |

---

## 실행 방법 (Google Colab 기준)

### 1단계 — 데이터셋 생성
```bash
# Dataset_preparation.py 실행
# 사전 조건: Google Drive에 knife_videos/ 폴더와 영상 파일 존재
```

### 2단계 — 모델 학습 및 평가
각 모델 스크립트를 Colab에서 순서대로 실행합니다.

```python
# GitHub에서 데이터셋 자동 다운로드
# 모델 학습 (10 epochs, ImageNet 사전학습 가중치)
# 조건별 정확도 분석 및 시각화
```

### 의존성 패키지
```
tensorflow
ultralytics
opencv-python
scikit-learn
matplotlib
pandas
```

---

## 분석 항목

각 모델 스크립트는 학습 후 아래 항목을 자동으로 분석·시각화합니다.

- **학습/검증 정확도 곡선** (Accuracy per Epoch)
- **Confusion Matrix**
- **조도별 정확도** (bright vs dark)
- **배경별 정확도** (kitchen, desk, livingroom, floor, window)
- **오클루전 여부별 정확도** (yes vs none)
- **칼 종류별 정확도**
- **폴더(영상 단위) 별 정확도** — 전체 조건 조합 분석

결과는 각각 `{model}_results.csv`로 저장됩니다.

---

## 모델 구조 (공통)

모든 모델은 동일한 전이학습 방식으로 구성됩니다.

```
ImageNet 사전학습 백본 (frozen)
  → GlobalAveragePooling2D
  → Dense(128, relu)
  → Dense(3, softmax)   # 식칼 / 과도 / 커터칼
```

- 옵티마이저: `Adam`
- 손실 함수: `categorical_crossentropy`
- 입력 크기: `224×224`
- 배치 크기: `32`
- 에폭: `10`
- Train/Val 분할: `80/20`

---

## 데이터셋 링크

학습에 사용된 이미지 데이터셋은 아래 GitHub 저장소에서 다운로드할 수 있습니다.

```
https://github.com/jwdebbie/knife-classification/raw/main/knives_dataset.zip
```
