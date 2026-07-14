import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from pathlib import Path
from ultralytics import YOLO
import hydra
from omegaconf import DictConfig


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=128, num_layers=2, num_classes=3):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def align_kps_yolo_to_dataset(yolo_kps):
    target_kps = np.zeros((16, 2))
    
    target_kps[0] = yolo_kps[16]
    target_kps[1] = yolo_kps[14]
    target_kps[2] = yolo_kps[12]
    target_kps[3] = yolo_kps[11]
    target_kps[4] = yolo_kps[13]
    target_kps[5] = yolo_kps[15]
    target_kps[6] = (yolo_kps[11] + yolo_kps[12]) / 2.0
    target_kps[7] = (yolo_kps[5] + yolo_kps[6]) / 2.0
    target_kps[8] = (yolo_kps[5] + yolo_kps[6]) / 2.0
    
    head_offset = (yolo_kps[11] - yolo_kps[5]) * 0.2
    target_kps[9] = yolo_kps[0] - head_offset
    
    target_kps[10] = yolo_kps[10]
    target_kps[11] = yolo_kps[8]
    target_kps[12] = yolo_kps[6]
    target_kps[13] = yolo_kps[5]
    target_kps[14] = yolo_kps[7]
    target_kps[15] = yolo_kps[9]
    
    return target_kps


def normalize_keypoints(kps_16, bbox):
    x_min, y_min, x_max, y_max = bbox
    w = x_max - x_min
    h = y_max - y_min
    
    if w == 0:
        w = 1.0
    if h == 0:
        h = 1.0
        
    normalized = []
    for kp in kps_16:
        x_norm = (kp[0] - x_min) / w
        y_norm = (kp[1] - y_min) / h
        normalized.extend([x_norm, y_norm])
        
    return normalized


def process_single_source(video_source, cfg, yolo_model, lstm_model, device):
    track_queues = {}
    prediction_history = {}
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_source}")
        return
        
    print(f"\nProcessing video: {video_source}")
    
    class_labels = ["Drinking", "Phone", "Smoking"]
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    
    step = int(fps / 20.0) if fps > 45 else int(fps / 10.0)
    step = max(1, step)
    print(f"Video FPS: {fps:.1f} | Frame extraction step: {step}")
    
    out_writer = None
    output_file_path = None
    if cfg.save_video and video_source != 0:
        video_name = Path(video_source).name if isinstance(video_source, str) else "webcam.mp4"
        resolved_output_path = cfg.output_path.replace("{video_name}", video_name).replace("${video.split[-1]}", video_name)
        
        orig_cwd = Path(hydra.utils.get_original_cwd())
        output_file_path = (orig_cwd / resolved_output_path).resolve()
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        save_fps = fps / step
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(str(output_file_path), fourcc, save_fps, (width, height))
        print(f"Video writer initialized: {output_file_path} (FPS: {save_fps:.2f}, Size: {width}x{height})")
    
    headless_mode = False
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % step != 0:
            continue
            
        results = yolo_model.track(frame, persist=True, tracker=cfg.tracker, verbose=False)
        
        if len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            keypoints_all = results[0].keypoints.xy.cpu().numpy()
            
            for bbox, track_id, yolo_kps in zip(boxes, track_ids, keypoints_all):
                kps_16 = align_kps_yolo_to_dataset(yolo_kps)
                norm_kps = normalize_keypoints(kps_16, bbox)
                
                if track_id not in track_queues:
                    track_queues[track_id] = deque(maxlen=cfg.seq_len)
                track_queues[track_id].append(norm_kps)
                
                probs = [0.0, 0.0, 0.0]
                
                if len(track_queues[track_id]) == cfg.seq_len:
                    seq_data = list(track_queues[track_id])
                    seq_tensor = torch.tensor([seq_data], dtype=torch.float32).to(device)
                    
                    with torch.no_grad():
                        logits = lstm_model(seq_tensor)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
                if track_id not in prediction_history:
                    prediction_history[track_id] = deque(maxlen=cfg.history_len)
                prediction_history[track_id].append(probs)
                
                avg_probs = np.mean(list(prediction_history[track_id]), axis=0)
                
                max_idx = np.argmax(avg_probs)
                max_prob = avg_probs[max_idx]
                current_action = class_labels[max_idx]
                
                if max_idx == 2 and max_prob > cfg.thresholds.smoking:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                    
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label_text = f"ID: {track_id} | {current_action}: {max_prob*100:.1f}%"
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                skeleton_connections = [
                    (0, 1), (0, 2), (1, 3), (2, 4),
                    (5, 6),
                    (5, 7), (7, 9), (6, 8), (8, 10),
                    (5, 11), (6, 12), (11, 12),
                    (11, 13), (13, 15), (12, 14), (14, 16)
                ]
                
                for pt1_idx, pt2_idx in skeleton_connections:
                    pt1 = yolo_kps[pt1_idx]
                    pt2 = yolo_kps[pt2_idx]
                    x_c1, y_c1 = int(pt1[0]), int(pt1[1])
                    x_c2, y_c2 = int(pt2[0]), int(pt2[1])
                    if (x_c1 > 0 or y_c1 > 0) and (x_c2 > 0 or y_c2 > 0):
                        cv2.line(frame, (x_c1, y_c1), (x_c2, y_c2), (255, 255, 0), 1)
                
                for kp in yolo_kps:
                    x_c, y_c = int(kp[0]), int(kp[1])
                    if x_c > 0 or y_c > 0:
                        cv2.circle(frame, (x_c, y_c), 3, (0, 255, 255), -1)
                
        if out_writer is not None:
            out_writer.write(frame)
            
        if not headless_mode:
            try:
                cv2.imshow("Smart Smoking Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except cv2.error:
                print("Warning: cv2.imshow is not supported in this environment. Saving to file only.")
                headless_mode = True
            
    cap.release()
    if out_writer is not None:
        out_writer.release()
        print(f"Output video successfully saved to {output_file_path}")
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


@hydra.main(version_base=None, config_path="configs", config_name="inference_config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    lstm_model = LSTMClassifier(
        input_dim=cfg.input_dim, 
        hidden_dim=cfg.hidden_dim, 
        num_layers=cfg.num_layers, 
        num_classes=cfg.num_classes
    ).to(device)
    
    orig_cwd = Path(hydra.utils.get_original_cwd())
    lstm_path = (orig_cwd / cfg.lstm_path).resolve()
    if lstm_path.exists():
        lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
        print(f"Loaded LSTM weights from {lstm_path}")
    else:
        print(f"Warning: {lstm_path} not found.")
    lstm_model.eval()
    
    print(f"Loading YOLOv8-Pose model from {cfg.yolo_model}...")
    yolo_model = YOLO(cfg.yolo_model)
    
    video_source = cfg.video if cfg.video else 0
    
    if isinstance(video_source, str):
        source_path = (orig_cwd / video_source).resolve()
        if source_path.is_dir():
            print(f"Scanning directory for video files: {source_path}")
            valid_exts = {".mp4", ".avi", ".mkv", ".mov", ".MP4", ".AVI", ".MKV", ".MOV"}
            video_files = [str(f) for f in source_path.iterdir() if f.is_file() and f.suffix in valid_exts]
            video_files.sort()
            print(f"Found {len(video_files)} video files in folder.")
            for vf in video_files:
                process_single_source(vf, cfg, yolo_model, lstm_model, device)
            return
        else:
            video_source = str(source_path)
            
    process_single_source(video_source, cfg, yolo_model, lstm_model, device)


if __name__ == "__main__":
    main()
