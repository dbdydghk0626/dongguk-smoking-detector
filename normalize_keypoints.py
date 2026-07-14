import os
import json
from pathlib import Path

def normalize_json_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process annotations
    if 'annotations' in data and isinstance(data['annotations'], list):
        for ann in data['annotations']:
            if 'bbox' in ann and 'keypoints' in ann:
                bbox = ann['bbox']
                keypoints = ann['keypoints']
                
                # Check bbox validity [x_min, y_min, x_max, y_max]
                if isinstance(bbox, list) and len(bbox) == 4:
                    x_min, y_min, x_max, y_max = bbox
                    w = x_max - x_min
                    h = y_max - y_min
                    
                    # Avoid division by zero
                    if w == 0:
                        w = 1.0
                    if h == 0:
                        h = 1.0
                    
                    normalized_kps = []
                    # Process each (x, y, v) triplet
                    for i in range(0, len(keypoints), 3):
                        if i + 2 < len(keypoints):
                            x = keypoints[i]
                            y = keypoints[i+1]
                            v = keypoints[i+2]
                            
                            x_norm = (x - x_min) / w
                            y_norm = (y - y_min) / h
                            
                            normalized_kps.extend([round(x_norm, 6), round(y_norm, 6), v])
                    
                    ann['keypoints'] = normalized_kps

    # Save to output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    # Find any non-hidden directory to avoid source encoding issues (starts with non-dot)
    dirs = [d for d in os.listdir(".") if os.path.isdir(d) and not d.startswith(".")]
    if not dirs:
        print("Error: Could not find data directory.")
        return
    
    korean_dir = dirs[0]
    base_dir = Path(os.path.abspath(korean_dir)) / "annotation"
    input_dir = base_dir / "Annotation_2D_tar"
    output_dir = base_dir / "Annotation_2D_normalized"
    
    print(f"Searching for JSON files in: {input_dir}")
    json_files = list(input_dir.glob("**/*.json"))
    total_files = len(json_files)
    print(f"Total JSON files found: {total_files}")
    
    processed_count = 0
    for file_path in json_files:
        # Determine relative path
        rel_path = file_path.relative_to(input_dir)
        target_path = output_dir / rel_path
        
        try:
            normalize_json_file(file_path, target_path)
            processed_count += 1
            if processed_count % 100 == 0 or processed_count == total_files:
                print(f"Processed {processed_count}/{total_files} files...")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print("Normalization completed successfully!")

if __name__ == "__main__":
    main()
