import cv2
import numpy as np
import onnxruntime as ort
from collections import deque
from pathlib import Path
import argparse
import time


YOLO_INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
CLASS_LABELS = ["Drinking", "Phone", "Smoking"]
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]


def letterbox(img, new_shape=640):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape - new_unpad[0]) / 2
    dh = (new_shape - new_unpad[1]) / 2

    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=(114, 114, 114))

    if img_padded.shape[0] != new_shape or img_padded.shape[1] != new_shape:
        img_padded = cv2.resize(img_padded, (new_shape, new_shape))

    return img_padded, r, (dw, dh)


def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms(boxes, scores, iou_threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep)


def postprocess_yolo(output, ratio, pad, orig_shape):
    predictions = output[0].T

    scores = predictions[:, 4]
    mask = scores > CONF_THRESHOLD
    predictions = predictions[mask]
    scores = scores[mask]

    if len(predictions) == 0:
        return [], [], []

    boxes = xywh2xyxy(predictions[:, :4])
    keypoints = predictions[:, 5:].reshape(-1, 17, 3)

    keep = nms(boxes, scores, IOU_THRESHOLD)
    boxes = boxes[keep]
    scores = scores[keep]
    keypoints = keypoints[keep]

    dw, dh = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_shape[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_shape[0])

    keypoints[:, :, 0] = (keypoints[:, :, 0] - dw) / ratio
    keypoints[:, :, 1] = (keypoints[:, :, 1] - dh) / ratio

    return boxes, scores, keypoints


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
    w = max(x_max - x_min, 1.0)
    h = max(y_max - y_min, 1.0)
    normalized = []
    for kp in kps_16:
        normalized.extend([(kp[0] - x_min) / w, (kp[1] - y_min) / h])
    return normalized


def simple_track(prev_boxes, prev_ids, new_boxes, next_id, dist_threshold=100):
    if len(prev_boxes) == 0 or len(new_boxes) == 0:
        new_ids = list(range(next_id, next_id + len(new_boxes)))
        return new_ids, next_id + len(new_boxes)

    prev_centers = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in prev_boxes])
    new_centers = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in new_boxes])

    assigned_ids = [-1] * len(new_boxes)
    used_prev = set()

    for i, nc in enumerate(new_centers):
        dists = np.linalg.norm(prev_centers - nc, axis=1)
        order = np.argsort(dists)
        for j in order:
            if dists[j] < dist_threshold and j not in used_prev:
                assigned_ids[i] = prev_ids[j]
                used_prev.add(j)
                break

    for i in range(len(assigned_ids)):
        if assigned_ids[i] == -1:
            assigned_ids[i] = next_id
            next_id += 1

    return assigned_ids, next_id


def run_inference(args):
    yolo_session = ort.InferenceSession(args.yolo_model, providers=['CPUExecutionProvider'])
    lstm_session = ort.InferenceSession(args.lstm_model, providers=['CPUExecutionProvider'])

    yolo_input_name = yolo_session.get_inputs()[0].name
    lstm_input_name = lstm_session.get_inputs()[0].name

    try:
        video_source = int(args.video)
        video_files = [video_source]
    except ValueError:
        video_dir = Path(args.video)
        if video_dir.is_dir():
            valid_exts = {".mp4", ".avi", ".mkv", ".mov"}
            video_files = sorted([f for f in video_dir.iterdir() if f.suffix.lower() in valid_exts])
            print(f"Found {len(video_files)} video files in {video_dir}")
        else:
            video_files = [str(video_dir)]

    for video_source in video_files:
        print(f"\nProcessing: {video_source}")
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open {video_source}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = int(fps / 20.0) if fps > 45 else int(fps / 10.0)
        step = max(1, step)

        out_writer = None
        if args.save_video:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            video_name = Path(str(video_source)).name if not isinstance(video_source, int) else "webcam.mp4"
            output_path = out_dir / video_name
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            save_fps = fps / step
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(str(output_path), fourcc, save_fps, (width, height))

        track_queues = {}
        prediction_history = {}
        prev_boxes = []
        prev_ids = []
        next_id = 0
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % step != 0:
                continue

            orig_shape = frame.shape
            img, ratio, pad = letterbox(frame, YOLO_INPUT_SIZE)
            img_input = img.astype(np.float32) / 255.0
            img_input = img_input.transpose(2, 0, 1)[np.newaxis, ...]

            yolo_output = yolo_session.run(None, {yolo_input_name: img_input})
            boxes, scores, keypoints = postprocess_yolo(yolo_output, ratio, pad, orig_shape)

            if len(boxes) > 0:
                track_ids, next_id = simple_track(prev_boxes, prev_ids, boxes, next_id)
                prev_boxes = boxes.copy()
                prev_ids = track_ids

                for bbox, track_id, kps in zip(boxes, track_ids, keypoints):
                    yolo_kps = kps[:, :2]
                    kps_16 = align_kps_yolo_to_dataset(yolo_kps)
                    norm_kps = normalize_keypoints(kps_16, bbox)

                    if track_id not in track_queues:
                        track_queues[track_id] = deque(maxlen=args.seq_len)
                    track_queues[track_id].append(norm_kps)

                    probs = [0.0, 0.0, 0.0]

                    if len(track_queues[track_id]) == args.seq_len:
                        seq = np.array([list(track_queues[track_id])], dtype=np.float32)
                        lstm_output = lstm_session.run(None, {lstm_input_name: seq})
                        logits = lstm_output[0][0]
                        exp_logits = np.exp(logits - np.max(logits))
                        probs = exp_logits / exp_logits.sum()

                    if track_id not in prediction_history:
                        prediction_history[track_id] = deque(maxlen=args.history_len)
                    prediction_history[track_id].append(probs)

                    avg_probs = np.mean(list(prediction_history[track_id]), axis=0)
                    max_idx = np.argmax(avg_probs)
                    max_prob = avg_probs[max_idx]
                    action = CLASS_LABELS[max_idx]

                    if max_idx == 2 and max_prob > args.smoking_threshold:
                        color = (0, 0, 255)
                        if args.enable_sound:
                            try:
                                import os
                                import sys
                                if sys.platform == "win32":  
                                    import winsound
                                    winsound.Beep(1000, 200)   
                                else:
                                    sys.stdout.write('\a')
                                    sys.stdout.flush()
                            except Exception:
                                pass
                    else:
                        color = (0, 255, 0)

                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID:{track_id} | {action}: {max_prob*100:.1f}%",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    for pt1_idx, pt2_idx in SKELETON_CONNECTIONS:
                        pt1 = yolo_kps[pt1_idx]
                        pt2 = yolo_kps[pt2_idx]
                        c1, c2 = (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1]))
                        if (c1[0] > 0 or c1[1] > 0) and (c2[0] > 0 or c2[1] > 0):
                            cv2.line(frame, c1, c2, (255, 255, 0), 1)

                    for kp in yolo_kps:
                        cx, cy = int(kp[0]), int(kp[1])
                        if cx > 0 or cy > 0:
                            cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

            if out_writer is not None:
                out_writer.write(frame)

            try:
                cv2.imshow("Smoking Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except cv2.error:
                pass

        cap.release()
        if out_writer is not None:
            out_writer.release()
            print(f"Saved: {output_path}")

    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_model", type=str, required=True)
    parser.add_argument("--lstm_model", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--history_len", type=int, default=15)
    parser.add_argument("--smoking_threshold", type=float, default=0.9)
    parser.add_argument("--enable_sound", action="store_true")
    parser.add_argument("--sound_path", type=str, default="warning.mp3")
    args = parser.parse_args()
    run_inference(args)
