import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
from tensorflow.keras import mixed_precision
import cv2
import numpy as np
from Anchor_box import Anchor_box
from RetinaNet import RetinaNet
from resnet_40 import resnet_40_backbone
from Xu_ly_du_lieu.tiling_utils import generate_tile_coords, apply_global_nms

# --- CAU HINH ---
TARGET_SIZE = 512
WEIGHT_PATH = '/home/huy/Documents/RetinaNet_DIoU/Drone Thermal.v4i.voc/weight_store/RetinaNet_anchorfree_6_7_2026_11:53.weights.h5'

INPUT_VIDEO  = '/home/huy/Documents/RetinaNet_DIoU/Drone Thermal.v4i.voc/video_drone/in_test_2.mp4'
OUTPUT_VIDEO = '/home/huy/Documents/RetinaNet_DIoU/Drone Thermal.v4i.voc/video_drone/out_test_2.mp4'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


def decode_box_predictions(anchor_boxes, box_predictions):
    anchor_boxes = tf.cast(anchor_boxes, box_predictions.dtype)
    a_x, a_y, a_w, a_h = tf.split(anchor_boxes, 4, axis=-1)
    t_x, t_y, t_w, t_h = tf.split(box_predictions, 4, axis=-1)

    t_x = t_x * 0.1
    t_y = t_y * 0.1
    t_w = t_w * 0.2
    t_h = t_h * 0.2

    x = t_x * a_w + a_x
    y = t_y * a_h + a_y
    w = tf.exp(t_w) * a_w
    h = tf.exp(t_h) * a_h

    x1, y1 = x - w / 2.0, y - h / 2.0
    x2, y2 = x + w / 2.0, y + h / 2.0

    return tf.concat([y1, x1, y2, x2], axis=-1)


def get_inference_model(num_classes):
    backbone = resnet_40_backbone()
    model = RetinaNet(num_classes=num_classes, backbone=backbone)
    model(tf.zeros((1, TARGET_SIZE, TARGET_SIZE, 3)))
    model.load_weights(WEIGHT_PATH)
    return model


def run_tiled_inference_on_frame(model, frame_bgr, all_anchors,
                                  tile_w=720, tile_h=576, overlap=0.22,
                                  score_threshold=0.5):
    """
    Nhan vao 1 frame BGR (numpy), chay tiled inference,
    tra ve boxes [ymin,xmin,ymax,xmax] trong he toa do goc va scores.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = tf.cast(tf.constant(frame_rgb), tf.float16)

    orig_h, orig_w = frame_bgr.shape[:2]
    tile_coords = generate_tile_coords(orig_w, orig_h, tile_w, tile_h, overlap)

    all_boxes = []
    all_scores = []

    for (tx, ty, tw, th) in tile_coords:
        tile = img[ty:ty + th, tx:tx + tw]

        tile_shape = tf.cast(tf.shape(tile)[:2], dtype=tf.float16)
        t_h_f, t_w_f = tile_shape[0], tile_shape[1]
        tile_ratio = tf.cast(TARGET_SIZE, tf.float16) / tf.math.maximum(t_h_f, t_w_f)

        new_h = tf.cast(t_h_f * tile_ratio, tf.int32)
        new_w = tf.cast(t_w_f * tile_ratio, tf.int32)

        tile_resized = tf.image.resize(tile, [new_h, new_w])
        tile_padded = tf.image.pad_to_bounding_box(tile_resized, 0, 0, TARGET_SIZE, TARGET_SIZE)
        tile_input = tf.expand_dims(tile_padded, axis=0)

        predictions = model.predict(tile_input, verbose=0)
        box_preds = predictions[:, :, :4]
        class_preds = predictions[:, :, 4:]

        decoded_boxes = decode_box_predictions(all_anchors, tf.cast(box_preds, tf.float32))
        class_probs = tf.nn.sigmoid(tf.cast(class_preds, tf.float32))

        nms_boxes, nms_scores, _, valid_det = tf.image.combined_non_max_suppression(
            tf.expand_dims(decoded_boxes, axis=2),
            class_probs,
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.3, # se loai bo nhung box trung lap nhieu hon
            score_threshold=score_threshold,
            clip_boxes=False
        )

        ratio_val = float(tile_ratio)
        n_det = int(valid_det[0])

        for i in range(n_det):
            ymin, xmin, ymax, xmax = [float(v) for v in nms_boxes[0][i]]
            all_boxes.append([
                ymin / ratio_val + ty,
                xmin / ratio_val + tx,
                ymax / ratio_val + ty,
                xmax / ratio_val + tx,
            ])
            all_scores.append(float(nms_scores[0][i]))

    final_boxes, final_scores, num_det = apply_global_nms(
        all_boxes, all_scores,
        iou_threshold=0.3,
        score_threshold=score_threshold
    )

    return final_boxes, final_scores, num_det


def draw_boxes(frame_bgr, boxes, scores, num_det):
    for i in range(num_det):
        ymin, xmin, ymax, xmax = int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])
        cv2.rectangle(frame_bgr, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
        label = f"human {scores[i]:.2f}"
        cv2.putText(frame_bgr, label, (xmin, max(ymin - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return frame_bgr


def calculate_iou(box1, box2):
    y1 = max(box1[0], box2[0])
    x1 = max(box1[1], box2[1])
    y2 = min(box1[2], box2[2])
    x2 = min(box1[3], box2[3])
    inter_area = max(0, y2 - y1) * max(0, x2 - x1)
    if inter_area == 0:
        return 0.0
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(box1_area + box2_area - inter_area)

class Track:
    def __init__(self, box, score):
        self.box = box
        self.score = score
        self.age = 0

if __name__ == '__main__':
    num_class = 1

    anchor_gene = Anchor_box()
    all_anchors = anchor_gene.get_anchors(img_h=TARGET_SIZE, img_w=TARGET_SIZE)
    model = get_inference_model(num_class)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Khong mo duoc video: {INPUT_VIDEO}")
        exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    OUT_FPS = 25
    # Chi lay 1 frame sau moi khoang nay de dat dung 25fps
    frame_step = max(1, round(src_fps / OUT_FPS))

    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, OUT_FPS, (width, height))

    print(f"Video goc: {width}x{height} @ {src_fps:.1f}fps, tong {total} frames")
    print(f"Output: {OUT_FPS}fps (lay 1 frame / {frame_step} frames)")
    print(f"Output: {OUTPUT_VIDEO}")

    frame_idx = 0
    tracks = []
    MAX_AGE = 5
    IOU_THRESHOLD = 0.2

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        if (frame_idx - 1) % frame_step != 0:
            continue

        print(f"Frame {frame_idx}/{total}", end='\r')

        boxes, scores, num_det = run_tiled_inference_on_frame(
            model, frame, all_anchors, score_threshold=0.5
        )

        new_tracks = []
        matched_det_indices = set()

        for track in tracks:
            best_iou = 0
            best_det_idx = -1
            for j in range(num_det):
                if j in matched_det_indices: continue
                iou = calculate_iou(track.box, boxes[j])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = j
                    
            if best_iou > IOU_THRESHOLD:
                track.box = boxes[best_det_idx]
                track.score = scores[best_det_idx]
                track.age = 0
                new_tracks.append(track)
                matched_det_indices.add(best_det_idx)
            else:
                track.age += 1
                if track.age <= MAX_AGE:
                    new_tracks.append(track)

        for j in range(num_det):
            if j not in matched_det_indices:
                new_tracks.append(Track(boxes[j], scores[j]))

        tracks = new_tracks

        final_boxes_to_draw = [t.box for t in tracks]
        final_scores_to_draw = [t.score for t in tracks]

        frame = draw_boxes(frame, final_boxes_to_draw, final_scores_to_draw, len(tracks))
        out.write(frame)

    cap.release()
    out.release()
    print(f"\nHoan thanh! Da luu video tai: {OUTPUT_VIDEO}")
