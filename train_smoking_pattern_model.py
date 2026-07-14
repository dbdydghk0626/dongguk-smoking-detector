import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt


class SmokingDataset(Dataset):
    def __init__(self, data_dir, seq_len=20):
        self.seq_len = seq_len
        self.sequences = []
        self.labels = []
        
        self.cat_to_label = {"41": 0, "42": 1, "45": 2}
        
        data_dir_path = Path(data_dir)
        print(f"Scanning normalized files in {data_dir_path}...")
        json_files = list(data_dir_path.glob("**/*.json"))
        print(f"Found {len(json_files)} files. Extracting sequences...")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'images' not in data or len(data['images']) == 0:
                    continue
                
                action_category = data['images'][0].get('action_category')
                if action_category not in self.cat_to_label:
                    continue
                label = self.cat_to_label[action_category]
                
                if 'annotations' not in data or not isinstance(data['annotations'], list):
                    continue
                    
                annotations = data['annotations']
                annotations.sort(key=lambda x: x['img_no'])
                
                frames_kps = []
                for ann in annotations:
                    if 'keypoints' in ann:
                        kps = ann['keypoints']
                        xy_kps = []
                        for i in range(0, len(kps), 3):
                            if i + 1 < len(kps):
                                xy_kps.extend([kps[i], kps[i+1]])
                        if len(xy_kps) == 32:
                            frames_kps.append(xy_kps)
                
                if len(frames_kps) >= self.seq_len:
                    for idx in range(len(frames_kps) - self.seq_len + 1):
                        seq = frames_kps[idx : idx + self.seq_len]
                        self.sequences.append(seq)
                        self.labels.append(label)
                        
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")
                
        self.sequences = torch.tensor(self.sequences, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        print(f"Total sequences extracted: {len(self.sequences)}")
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


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


@hydra.main(version_base=None, config_path="configs", config_name="train_config")
def main(cfg: DictConfig):
    orig_cwd = Path(hydra.utils.get_original_cwd())
    
    dirs = [d for d in os.listdir(orig_cwd) if os.path.isdir(orig_cwd / d) and not d.startswith(".")]
    if not dirs:
        print("Error: Could not find data directory.")
        return
        
    korean_dir = dirs[0]
    data_dir = orig_cwd / korean_dir / "annotation" / "Annotation_2D_normalized"
    
    save_dir = (orig_cwd / cfg.save_path).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = (orig_cwd / cfg.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard log directory: {log_dir}")
    
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist.")
        return
        
    dataset = SmokingDataset(data_dir, seq_len=cfg.seq_len)
    if len(dataset) == 0:
        print("Error: No training sequences found.")
        return
        
    train_size = int(cfg.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = LSTMClassifier(
        input_dim=cfg.input_dim, 
        hidden_dim=cfg.hidden_dim, 
        num_layers=cfg.num_layers, 
        num_classes=cfg.num_classes
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    epochs = cfg.epochs
    best_val_acc = 0.0
    best_train_acc = 0.0
    best_train_loss = float('inf')
    best_val_loss = float('inf')
    
    last_model_path = save_dir / "last_model.pth"
    best_train_loss_path = save_dir / "best_train_loss.pth"
    best_val_loss_path = save_dir / "best_val_loss.pth"
    best_train_acc_path = save_dir / "best_train_acc.pth"
    best_val_acc_path = save_dir / "best_val_acc.pth"
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs:02d} [Train]", leave=False)
        for inputs, targets in train_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()
            
            train_bar.set_postfix(loss=loss.item())
            
        train_loss /= len(train_dataset)
        train_acc = correct_train / total_train
        
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{epochs:02d} [Val]", leave=False)
        with torch.no_grad():
            for inputs, targets in val_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_val += targets.size(0)
                correct_val += predicted.eq(targets).sum().item()
                
                val_bar.set_postfix(loss=loss.item())
                
        val_loss /= len(val_dataset)
        val_acc = correct_val / total_val
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")
        
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch + 1)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch + 1)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        torch.save(model.state_dict(), last_model_path)
        
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), best_train_loss_path)
            print(f"  => Saved new best train loss model (Loss: {best_train_loss:.4f})")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_val_loss_path)
            print(f"  => Saved new best val loss model (Loss: {best_val_loss:.4f})")

        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save(model.state_dict(), best_train_acc_path)
            print(f"  => Saved new best train acc model (Acc: {best_train_acc*100:.2f}%)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_val_acc_path)
            print(f"  => Saved new best val acc model (Acc: {best_val_acc*100:.2f}%)")
            
    writer.close()
    
    try:
        epochs_range = range(1, epochs + 1)
        
        plt.figure(figsize=(10, 5))
        plt.plot(epochs_range, history["train_loss"], label="Train Loss", color="blue")
        plt.plot(epochs_range, history["val_loss"], label="Val Loss", color="red")
        plt.title("Train vs Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        loss_plot_path = save_dir / "loss_plot.png"
        plt.savefig(loss_plot_path, dpi=150)
        plt.close()
        print(f"Loss curve plot saved to: {loss_plot_path}")
        
        plt.figure(figsize=(10, 5))
        plt.plot(epochs_range, [acc * 100 for acc in history["train_acc"]], label="Train Acc", color="blue")
        plt.plot(epochs_range, [acc * 100 for acc in history["val_acc"]], label="Val Acc", color="red")
        plt.title("Train vs Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True)
        acc_plot_path = save_dir / "accuracy_plot.png"
        plt.savefig(acc_plot_path, dpi=150)
        plt.close()
        print(f"Accuracy curve plot saved to: {acc_plot_path}")
        
    except Exception as e:
        print(f"Error plotting history: {e}")
        
    print("Training finished successfully!")


if __name__ == "__main__":
    main()
