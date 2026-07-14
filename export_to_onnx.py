import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import torch
import torch.nn as nn
from pathlib import Path
import hydra
from omegaconf import DictConfig


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


@hydra.main(version_base=None, config_path="configs", config_name="onnx_config")
def main(cfg: DictConfig):
    device = torch.device("cpu")
    print(f"Exporting model to ONNX using device: {device}")
    
    model = LSTMClassifier(
        input_dim=cfg.input_dim, 
        hidden_dim=cfg.hidden_dim, 
        num_layers=cfg.num_layers, 
        num_classes=cfg.num_classes
    ).to(device)
    
    orig_cwd = Path(hydra.utils.get_original_cwd())
    lstm_path = (orig_cwd / cfg.lstm_path).resolve()
    if not lstm_path.exists():
        print(f"Error: Trained model weights not found at {lstm_path}")
        return
        
    model.load_state_dict(torch.load(lstm_path, map_location=device))
    model.eval()
    print(f"Loaded trained weights from {lstm_path}")
    
    dummy_input = torch.randn(1, cfg.seq_len, cfg.input_dim, device=device)
    
    onnx_output_path = (orig_cwd / cfg.onnx_path).resolve()
    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Exporting LSTM model to ONNX format...")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_output_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"\nLSTM ONNX model exported to: {onnx_output_path}")
    
    print(f"\nLoading YOLOv8-Pose model for ONNX export: {cfg.yolo_model}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from ultralytics import YOLO
    try:
        yolo_model = YOLO(cfg.yolo_model)
        print("Exporting YOLOv8-Pose model to ONNX format...")
        yolo_onnx_path = yolo_model.export(format="onnx", opset=12, simplify=True)
        print(f"YOLOv8-Pose ONNX model exported to: {yolo_onnx_path}")
    except Exception as e:
        print(f"Warning: YOLOv8-Pose ONNX export failed: {e}")


if __name__ == "__main__":
    main()
