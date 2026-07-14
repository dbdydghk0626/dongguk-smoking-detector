# Smart Smoking Detector
**동국대학교 교수학습혁신센터 주관 「2025 창의문제해결 프로젝트」 최종 상위 10개 팀 선정 프로젝트 **

CCTV 영상에서 사람의 신체 관절(skeleton) 좌표를 실시간으로 추출하고, 일정 프레임 구간의 관절 움직임 패턴을 LSTM으로 분류하여 금연 구역 내 흡연 행위를 자동 감지하는 시스템입니다.

## 🏫 Project Background 

본교 신공학관 3층 출입구 앞 금연구역은 상습적인 흡연 행위로 인해 학내 구성원들에게 오랜 기간 불편을 초래해 왔습니다. 현재 설치된 금연 안내 현수막은 사실상 안내·경고 기능을 상실하였고, 이는 단순한 시각적 수단만으로는 흡연 억제에 한계가 있음을 보여줍니다. 이에 공간(동선·환경), 행태(심리적 개입), 기술(감지)을 통합 설계하여 흡연자가 자발적으로 공식 흡연구역으로 이동하도록 유도하는 솔루션을 개발하였습니다.

### 👥 팀원 구성 및 역할 분담 (Team Members)

| **유용화** | **팀장** / AI 시스템 개발| LSTM 기반 2단계 실시간 흡연 감지 파이프라인 개발 |
| **서다경** | 공간 디자인 기획| CPTED(범죄예방환경설계) 및 자연적 감시 원리를 적용한 금연구역 환경설계 구상 |
| **양지원** | 분석 및 경제성 평가| 유도 구조물 인간공학적 접근 분석 및 설치 경제성 평가 수행 |
| **이상영** | 조사 및 분석 총괄| 오프라인 배너 설치, 온·오프라인 설문조사(163명) 및 대면 인터뷰 분석 |
| **이효석** | 시각화 담당| 개선 금연구역·흡연부스 공간 3D 시각화 및 시뮬레이션 구현 |

## 🔍 Overview

CCTV 영상에서 사람의 신체 관절(skeleton) 좌표를 실시간으로 추출하고, 일정 프레임 구간의 관절 움직임 패턴을 LSTM으로 분류하여 흡연 행위를 자동 감지합니다.

### Pipeline

```
Video Input → YOLOv8-Pose (관절 추출) → ByteTrack (다중 인물 추적) → LSTM (행동 분류) → 흡연 감지
```

### Action Classes

| Class ID | Action | Description |
|----------|--------|-------------|
| 0 | Drinking | 음료 마시기 |
| 1 | Phone | 핸드폰 조작 |
| 2 | Smoking | 흡연 |

## Project Structure

```
dongguk-smoking-detector/
├── configs/
│   ├── train_config.yaml
│   ├── inference_config.yaml
│   └── onnx_config.yaml
├── train_smoking_pattern_model.py
├── main_detector_yolopose.py
├── export_to_onnx.py
├── inference_raspi.py
└── README.md
```

## Requirements

```
torch
ultralytics
opencv-python
numpy
hydra-core
omegaconf
tqdm
tensorboard
matplotlib
onnx
onnxruntime
```

## Usage

### 1. Train

```bash
python train_smoking_pattern_model.py
```

- 정규화된 2D 관절 좌표 JSON 데이터를 슬라이딩 윈도우 방식으로 시퀀스화
- Train/Val 분할 후 LSTM 학습 수행
- TensorBoard 로깅 및 Loss/Accuracy 그래프 자동 저장
- best_val_acc, best_val_loss, best_train_acc, best_train_loss, last_model 총 5개 체크포인트 저장

### 2. Inference

```bash
python main_detector_yolopose.py
```

- 단일 비디오 파일 또는 폴더(디렉토리) 경로 지정 가능
- 폴더 지정 시 내부 모든 비디오 파일(.mp4, .avi, .mkv, .mov) 순차 처리
- 실시간 스켈레톤 오버레이 및 행동 분류 결과 시각화
- 흡연 감지 시 빨간색 바운딩 박스 경고, 정상 행동 시 초록색 표시
- GUI 미지원 환경(headless)에서도 비디오 파일로 자동 저장

### 3. ONNX Export

```bash
python export_to_onnx.py
```

- LSTM 분류 모델과 YOLOv8-Pose 모델을 각각 ONNX 포맷으로 변환
- 라즈베리파이 등 임베디드 환경에서 onnxruntime으로 추론 가능

### 4. Raspberry Pi Inference

```bash
python inference_raspi.py \
    --yolo_model
    --lstm_model 
    --video 
    --save_video 
    --output_dir output 
    --smoking_threshold 
```

- PyTorch, Ultralytics 없이 **onnxruntime + OpenCV + NumPy** 만으로 동작
- ONNX 변환된 YOLOv8-Pose + LSTM 모델 사용
- 경량 centroid 기반 트래커 내장 (ByteTrack 의존성 제거)
- 라즈베리파이 필요 패키지: `pip install onnxruntime opencv-python-headless numpy`

## Configuration

### inference_config.yaml

| Parameter | Description |
|-----------|-------------|
| `yolo_model` | YOLOv8-Pose 가중치 경로 |
| `lstm_path` | LSTM 분류기 가중치 경로 |
| `tracker` | 다중 인물 추적기 (bytetrack.yaml) |
| `seq_len` | LSTM 입력 시퀀스 길이 |
| `history_len` | 예측 결과 시간 평활화 윈도우 크기 |
| `video` | 비디오 폴더 경로 |
| `save_video` | 결과 비디오 저장 여부 |
| `thresholds.smoking` | 흡연 경고 확률 임계값 설정|

### train_config.yaml

| Parameter | Description |
|-----------|-------------|
| `hidden_dim` | LSTM Hidden 차원 |
| `num_layers` | LSTM 레이어 수 |
| `epochs` | 학습 에폭 수 |
| `batch_size` | 배치 크기 |
| `lr` | 학습률 |
| `train_split` | Train/Val 분할 비율 |

## Model Architecture

### YOLOv8n-Pose
- 81 layers, 3.29M parameters, 9.2 GFLOPs
- COCO Keypoint 17개 관절 추출
- ByteTrack 기반 다중 인물 추적

### LSTM Classifier
- Input: 32차원 (16개 관절 × 2 좌표)
- Hidden: 128차원, 2 layers
- Dropout: 0.2
- Output: 3 classes (Drinking, Phone, Smoking)
- Temporal Smoothing 적용 (최근 15프레임 예측 평균)
