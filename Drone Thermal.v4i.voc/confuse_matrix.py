import sys
import os

# Thêm thư mục gốc vào sys.path để Python tìm được module Anchor_box, RetinaNet, ...
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from predict_drone import get_inference_model, run_tiled_inference, TARGET_SIZE, TEST_CSV
# pyrefly: ignore [missing-import]
from Anchor_box import Anchor_box


def calculate_iou(box1, box2):
    """Tính IoU cho 2 hộp [xmin, ymin, xmax, ymax]"""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection_area / float(box1_area + box2_area - intersection_area + 1e-8)


def generate_confusion_matrix():
    print("--- Đang khởi tạo Model để tạo Confusion Matrix ---")
    test_df = pd.read_csv(TEST_CSV)

    # Lấy thông tin class
    all_labels = []
    for cid in test_df['class_id'].apply(ast.literal_eval):
        all_labels.extend(cid)
    unique_classes = sorted(list(set(all_labels)))
    num_class = len(unique_classes)

    # Map class_id sang index (0, 1, 2...)
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}

    anchor_gene = Anchor_box()
    all_anchors = anchor_gene.get_anchors(img_h=TARGET_SIZE, img_w=TARGET_SIZE)
    model = get_inference_model(num_class)

    # Ma trận: (Số class + 1) để tính cả Background (nếu bỏ sót hoặc đoán nhầm vào nền)
    # Rows: Ground Truth, Cols: Predictions
    matrix = np.zeros((num_class + 1, num_class + 1), dtype=int)

    print(f"Đang xử lý {len(test_df)} ảnh...")

    for index, row in test_df.iterrows():
        img_path = row['path_img']
        gt_boxes_raw = ast.literal_eval(row['bbox'])
        gt_classes_raw = ast.literal_eval(row['class_id'])

        # Chạy inference (Dùng ngưỡng 0.5 để ma trận phản ánh thực tế sử dụng)
        _, pred_boxes, pred_scores, _, num_det, ratio = run_tiled_inference(
            model, img_path, all_anchors, score_threshold=0.5
        )

        # Chuẩn bị GT
        gt_boxes = []
        for box in gt_boxes_raw:
            x_c, y_c, w, h = box
            xmin = (x_c - w / 2.0) * ratio
            ymin = (y_c - h / 2.0) * ratio
            xmax = (x_c + w / 2.0) * ratio
            ymax = (y_c + h / 2.0) * ratio
            gt_boxes.append([xmin, ymin, xmax, ymax])

        matched_gt = [False] * len(gt_boxes)

        # 1. Kiểm tra True Positives (TP) và False Positives (FP)
        for i in range(num_det):
            ymin_p, xmin_p, ymax_p, xmax_p = pred_boxes[i]
            p_box = [xmin_p, ymin_p, xmax_p, ymax_p]

            best_iou = 0
            best_gt_idx = -1
            for j, g_box in enumerate(gt_boxes):
                iou = calculate_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= 0.5 and best_gt_idx != -1 and not matched_gt[best_gt_idx]:
                # Khớp đúng (Giả sử bạn hiện tại chỉ có 1 class hoặc class_id cố định)
                # Ở đây mình lấy class_id đầu tiên của ảnh đó (Huy có thể chỉnh nếu bài toán đa lớp)
                c_idx = class_to_idx[gt_classes_raw[best_gt_idx]]
                matrix[c_idx, c_idx] += 1
                matched_gt[best_gt_idx] = True
            else:
                # Dự đoán sai hoặc vào nền (False Positive)
                # Cột cuối cùng đại diện cho Background trong thực tế nhưng dự đoán ra vật thể
                matrix[num_class, 0] += 1  # Đánh dấu vào hàng Background

        # 2. Kiểm tra False Negatives (FN - Bỏ sót vật thể)
        for j, matched in enumerate(matched_gt):
            if not matched:
                c_idx = class_to_idx[gt_classes_raw[j]]
                matrix[c_idx, num_class] += 1  # Cột cuối cùng là Background (Model không thấy gì)

    # --- LƯU KẾT QUẢ ---
    base_dir = '/home/huy/Documents/RetinaNet_FreeAnchor_DIoU/Drone Thermal.v4i.voc'
    output_path = os.path.join(base_dir, 'recall_precision_csv/RetinaNet_anchorfree_DIoU')
    os.makedirs(output_path, exist_ok=True)

    # Tên các nhãn (Thêm Background)
    labels = [str(c) for c in unique_classes] + ["Background"]

    # Lưu CSV
    df_cm = pd.DataFrame(matrix, index=labels, columns=labels)
    df_cm.to_csv(os.path.join(output_path, 'confusion_matrix.csv'))

    # Vẽ đồ thị
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual (Ground Truth)')
    plt.title('Confusion Matrix (Tiled Inference)')

    plot_file = os.path.join(output_path, 'confusion_matrix.png')
    plt.savefig(plot_file, dpi=300)
    plt.close()

    print("--------------------------------------------------")
    print(f"Hoàn thành! Đã lưu Confusion Matrix tại: {output_path}")
    print("--------------------------------------------------")


if __name__ == '__main__':
    generate_confusion_matrix()