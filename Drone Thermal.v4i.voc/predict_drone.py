"""
this file use for datasets that dont have ground truth.
file du doan nay dung cho du doan loai du lieu co nhan.
"""
import sys
import os
# Thêm thư mục gốc của dự án vào sys.path để Python tìm được các module:
# Anchor_box, RetinaNet, resnet_40, Xu_ly_du_lieu
# (Cần thiết vì file này nằm trong thư mục con 'Drone Thermal.v4i.voc/')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Anchor_box import Anchor_box
from RetinaNet import RetinaNet
from resnet_40 import resnet_40_backbone
import ast

# [TILING - THÊM MỚI] import các hàm tiling từ tiling_utils
from Xu_ly_du_lieu.tiling_utils import generate_tile_coords, apply_global_nms
from tensorflow.keras import mixed_precision
# --- CAU HINH ---
TARGET_SIZE = 512   # kích thước mỗi tile sau khi resize (không đổi)
BATCH_SIZE = 1
# [TILING - ĐỔI] trỏ sang weight của model mới train với tiled data
WEIGHT_PATH = '/home/huy/Documents/RetinaNet_FreeAnchor_DIoU/Drone Thermal.v4i.voc/weight_store/RetinaNet_anchorfree_DIoU.weights.h5'
TEST_CSV = '/home/huy/Documents/RetinaNet_FreeAnchor_DIoU/Drone Thermal.v4i.voc/csv_file/test_data_6_7_2026.csv'
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
    """
    Giai ma toa do tu delta sang [ymin, xmin, ymax, xmax] cho NMS
    """
    # Tach toa do Anchor (x_center, y_center, w, h)
    anchor_boxes = tf.cast(anchor_boxes, box_predictions.dtype)
    a_x, a_y, a_w, a_h = tf.split(anchor_boxes, 4, axis=-1)

    # Tach Delta du doan
    t_x, t_y, t_w, t_h = tf.split(box_predictions, 4, axis=-1)

    # [BAN VA 1]: Nhan nguoc lai voi variances da nen luc train
    # Phai trung khop voi file LabelEncoder luc train nhe bro
    t_x = t_x * 0.1
    t_y = t_y * 0.1
    t_w = t_w * 0.2
    t_h = t_h * 0.2

    # Giai ma Center X, Y
    x = t_x * a_w + a_x
    y = t_y * a_h + a_y

    # Giai ma Width, Height
    w = tf.exp(t_w) * a_w
    h = tf.exp(t_h) * a_h

    # Chuyen ve dang goc [xmin, ymin, xmax, ymax]
    x1, y1 = x - w / 2.0, y - h / 2.0
    x2, y2 = x + w / 2.0, y + h / 2.0

    # Tra ve y-first cho ham combined_non_max_suppression cua TF
    return tf.concat([y1, x1, y2, x2], axis=-1)


def get_inference_model(num_classes):
    backbone = resnet_40_backbone()
    model = RetinaNet(num_classes=num_classes, backbone=backbone)
    # Khoi tao shape cho model truoc khi load weight
    model(tf.zeros((1, TARGET_SIZE, TARGET_SIZE, 3)))
    model.load_weights(WEIGHT_PATH)
    return model
# [TILING - HÀM MỚI] ============================================================
def run_tiled_inference(model, image_path, all_anchors,
                        tile_w=640, tile_h=512, overlap=0.20,
                        score_threshold=0.7):
    """
    Chạy inference trên ảnh gốc bằng cách chia thành các tile, inference từng tile,
    gộp lại và áp dụng Global NMS để loại bỏ box trùng ở vùng overlap.

    Trả về boxes trong hệ tọa độ ảnh gốc (không cần scale lại bằng ratio).
    ratio trả về = 1.0 để tương thích với code vẽ GT box trong __main__.

    Args:
        model         : RetinaNet model đã load weights
        image_path    : đường dẫn ảnh gốc (1280×1024)
        all_anchors   : anchor boxes cho TARGET_SIZE×TARGET_SIZE
        tile_w, tile_h: kích thước mỗi tile (pixel trong ảnh gốc)
        overlap       : tỉ lệ chồng lấp giữa các tile (0.0 – 1.0)
        score_threshold: lọc box có confidence thấp hơn ngưỡng này

    Returns:
        (orig_img_uint8, final_boxes, final_scores, dummy_classes, num_det, ratio=1.0)
        final_boxes shape: (N, 4) — [ymin, xmin, ymax, xmax] trong hệ ảnh gốc
    """
    # Bước 1: Load ảnh gốc
    img_raw = tf.io.read_file(image_path)
    img     = tf.image.decode_jpeg(img_raw, channels=3)
    img     = tf.cast(img, tf.float32)
    orig_shape = tf.shape(img)
    orig_h = int(orig_shape[0])
    orig_w = int(orig_shape[1])

    # Bước 2: Tạo danh sách tọa độ tile
    tile_coords = generate_tile_coords(orig_w, orig_h, tile_w, tile_h, overlap)

    all_boxes  = []
    all_scores = []

    for (tx, ty, tw, th) in tile_coords:
        # Crop tile từ ảnh gốc
        tile = img[ty:ty + th, tx:tx + tw]

        # Resize và pad tile về TARGET_SIZE × TARGET_SIZE (giống resize_and_pad_img)
        tile_shape = tf.cast(tf.shape(tile)[:2], dtype=tf.float32)
        t_h, t_w   = tile_shape[0], tile_shape[1]
        tile_ratio = tf.cast(TARGET_SIZE, tf.float32) / tf.math.maximum(t_h, t_w)

        new_h = tf.cast(t_h * tile_ratio, tf.int32)
        new_w = tf.cast(t_w * tile_ratio, tf.int32)

        tile_resized = tf.image.resize(tile, [new_h, new_w])
        tile_padded  = tf.image.pad_to_bounding_box(tile_resized, 0, 0, TARGET_SIZE, TARGET_SIZE)
        tile_input   = tf.expand_dims(tile_padded, axis=0)

        # Chạy model trên tile
        predictions   = model.predict(tile_input, verbose=0)
        box_preds     = predictions[:, :, :4]
        class_preds   = predictions[:, :, 4:]

        decoded_boxes = decode_box_predictions(all_anchors, tf.cast(box_preds, tf.float32))
        class_probs   = tf.nn.sigmoid(tf.cast(class_preds, tf.float32))

        # NMS nội tile (lọc thô, threshold thấp để không bỏ sót)
        nms_boxes, nms_scores, _, valid_det = tf.image.combined_non_max_suppression(
            tf.expand_dims(decoded_boxes, axis=2),
            class_probs,
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.3,
            score_threshold=score_threshold,
            clip_boxes=False
        )

        ratio_val = float(tile_ratio)
        n_det     = int(valid_det[0])

        # Bước 3: Chuyển tọa độ boxes về hệ ảnh gốc
        #   model space (TARGET_SIZE) → tile gốc space (÷ ratio)
        #   tile gốc space            → ảnh gốc space (+ tile offset)
        for i in range(n_det):
            ymin, xmin, ymax, xmax = [float(v) for v in nms_boxes[0][i]]
            all_boxes.append([
                ymin / ratio_val + ty,   # ymin trong ảnh gốc
                xmin / ratio_val + tx,   # xmin trong ảnh gốc
                ymax / ratio_val + ty,   # ymax trong ảnh gốc
                xmax / ratio_val + tx,   # xmax trong ảnh gốc
            ])
            all_scores.append(float(nms_scores[0][i]))

    # Bước 4: Global NMS — BẮT BUỘC để loại box trùng ở vùng overlap
    final_boxes, final_scores, num_det = apply_global_nms(
        all_boxes, all_scores,
        iou_threshold=0.3,
        score_threshold=score_threshold
    )

    # Trả về ảnh gốc (uint8) và boxes trong hệ tọa độ ảnh gốc
    orig_img_uint8  = tf.cast(img, tf.uint8).numpy()
    dummy_classes   = np.zeros(num_det, dtype=np.float32)

    # ratio = 1.0 vì boxes đã ở hệ ảnh gốc → code vẽ GT trong __main__ hoạt động đúng
    return orig_img_uint8, final_boxes, final_scores, dummy_classes, num_det, 1.0
# [TILING - KẾT THÚC HÀM MỚI] ==================================================


if __name__ == '__main__':

    test_df = pd.read_csv(TEST_CSV)
    all_labels = []
    for cid in test_df['class_id'].apply(ast.literal_eval):
        all_labels.extend(cid)

    num_class = len(set(all_labels))

    anchor_gene = Anchor_box()
    all_anchors = anchor_gene.get_anchors(img_h=TARGET_SIZE, img_w=TARGET_SIZE)
    model = get_inference_model(num_class)

    for i in range(100):
        row = test_df.iloc[i]
        img_path = row['path_img']
        gt_boxes = ast.literal_eval(row['bbox'])

        # [TILING - ĐỔI] dùng run_tiled_inference thay cho run_inference
        # img giờ là ảnh gốc 1280×1024 (thay vì 512×512 padded)
        # boxes đã ở hệ tọa độ ảnh gốc, ratio = 1.0
        img, boxes, scores, classes, num_det, ratio = run_tiled_inference(
            model, img_path, all_anchors, score_threshold=0.7
        )

        print(f"--- Anh thu {i+1} ---")
        print(f"So luong vat the tim thay: {num_det}")
        if num_det > 0:
            # [TILING - ĐỔI] scores là numpy array, không gọi .numpy()
            print(f"Diem tin tuong (Scores): {scores[:num_det]}")

        fig, ax = plt.subplots(1, figsize=(8, 6))
        ax.imshow(img)

        for j in range(num_det):
            # [TILING - GIỮ NGUYÊN] boxes vẫn là [ymin, xmin, ymax, xmax],
            # nhưng giờ trong hệ ảnh gốc 1280×1024 thay vì 512×512
            ymin, xmin, ymax, xmax = float(boxes[j][0]), float(boxes[j][1]), float(boxes[j][2]), float(boxes[j][3])

            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='cyan',
                                     facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin - 5, f"Pred: {scores[j]:.2f}", color='cyan', fontsize=10, backgroundcolor='black')

        # Ve Ground Truth (Mau Do)
        # [TILING - GIỮ NGUYÊN] ratio = 1.0 nên x_c * ratio = x_c (tọa độ ảnh gốc)
        for box in gt_boxes:
            x_c, y_c, w, h = box
            x_c_scaled, y_c_scaled = x_c * ratio, y_c * ratio
            w_scaled, h_scaled = w * ratio, h * ratio

            rect_gt = patches.Rectangle((x_c_scaled - w_scaled / 2.0, y_c_scaled - h_scaled / 2.0),
                                        w_scaled, h_scaled, linewidth=1, edgecolor='red',
                                        facecolor='none', linestyle='--')
            ax.add_patch(rect_gt)

        plt.title(f"Test Image {i + 1} - Found {num_det} objects")
        plt.axis('off')
        plt.show()