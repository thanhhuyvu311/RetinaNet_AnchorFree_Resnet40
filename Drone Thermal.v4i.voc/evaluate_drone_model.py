import pandas as pd
import numpy as np
import ast
import os
import matplotlib.pyplot as plt
from predict_drone import get_inference_model, run_tiled_inference, TARGET_SIZE, TEST_CSV
# pyrefly: ignore [missing-import]
from Anchor_box import Anchor_box


def calculate_iou(box1, box2):
    """Tính IoU cho 2 hộp có dạng [xmin, ymin, xmax, ymax]"""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / float(box1_area + box2_area - intersection_area + 1e-8)
    return iou


def evaluate_model():
    print("--- Đang khởi tạo Model và Data để đánh giá ---")
    test_df = pd.read_csv(TEST_CSV)

    # Lấy số lượng class
    all_labels = []
    for cid in test_df['class_id'].apply(ast.literal_eval):
        all_labels.extend(cid)
    num_class = len(set(all_labels))

    anchor_gene = Anchor_box()
    all_anchors = anchor_gene.get_anchors(img_h=TARGET_SIZE, img_w=TARGET_SIZE)
    model = get_inference_model(num_class)

    all_predictions = []
    total_gt_boxes = 0

    # Giới hạn số lượng ảnh test nếu muốn chạy nhanh (ví dụ: test_df.head(100))
    print(f"Bắt đầu quét {len(test_df)} ảnh trong tập Test bằng cơ chế Tiling...")

    for index, row in test_df.iterrows():
        img_path = row['path_img']
        gt_boxes_raw = ast.literal_eval(row['bbox'])

        # Chạy dự đoán với score_threshold RẤT THẤP (0.05) để không bỏ sót box nào,
        # giúp vẽ PR Curve được đầy đủ và tính mAP chính xác hơn.
        # Lưu ý: score_threshold này là Ngưỡng Tự Tin (Confidence), khác hoàn toàn với IoU threshold!
        _, pred_boxes, pred_scores, _, num_det, ratio = run_tiled_inference(
            model, img_path, all_anchors, score_threshold=0.05
        )

        # Xử lý Ground Truth (Chuyển từ [x_c, y_c, w, h] -> [xmin, ymin, xmax, ymax])
        # Vì ratio = 1.0 nên tọa độ giữ nguyên ở hệ ảnh gốc
        gt_boxes = []
        for box in gt_boxes_raw:
            x_c, y_c, w, h = box
            x_c, y_c, w, h = x_c * ratio, y_c * ratio, w * ratio, h * ratio
            xmin = x_c - w / 2.0
            ymin = y_c - h / 2.0
            xmax = x_c + w / 2.0
            ymax = y_c + h / 2.0
            gt_boxes.append([xmin, ymin, xmax, ymax])
            total_gt_boxes += 1

        # Trạng thái khớp của GT trong ảnh này
        gt_matched = [False] * len(gt_boxes)

        # Lưu lại các hộp dự đoán để tính mAP chung
        for i in range(num_det):
            # pred_boxes ở dạng [ymin, xmin, ymax, xmax] -> Chuyển sang [xmin, ymin, xmax, ymax]
            ymin_p, xmin_p, ymax_p, xmax_p = pred_boxes[i]
            all_predictions.append({
                'image_idx': index,
                'box': [xmin_p, ymin_p, xmax_p, ymax_p],
                'score': pred_scores[i],
                'gt_boxes': gt_boxes,
                'gt_matched': gt_matched
            })

    # --- TÍNH TOÁN AP (Average Precision) ---
    print("\n--- Đang tính toán mAP ---")
    if total_gt_boxes == 0 or len(all_predictions) == 0:
        print("Lỗi: Không tìm thấy Ground Truth hoặc Dự đoán nào.")
        return

    # 1. Sắp xếp tất cả dự đoán toàn cục theo Score giảm dần
    all_predictions.sort(key=lambda x: x['score'], reverse=True)

    true_positives = np.zeros(len(all_predictions))
    false_positives = np.zeros(len(all_predictions))

    # 2. So khớp từng box dự đoán với GT
    for i, pred in enumerate(all_predictions):
        pred_box = pred['box']
        gt_boxes = pred['gt_boxes']
        gt_matched = pred['gt_matched']

        best_iou = 0.0
        best_gt_idx = -1

        for j, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        # Ngưỡng IoU chuẩn là 0.5
        if best_iou >= 0.5 and best_gt_idx != -1 and not gt_matched[best_gt_idx]:
            true_positives[i] = 1
            gt_matched[best_gt_idx] = True
        else:
            false_positives[i] = 1

    # 3. Tính Precision & Recall cộng dồn
    cum_tp = np.cumsum(true_positives)
    cum_fp = np.cumsum(false_positives)

    recalls = cum_tp / total_gt_boxes
    precisions = cum_tp / (cum_tp + cum_fp)

    # 4. Tính toán đường cong và AP (Pascal VOC)
    m_recalls = np.concatenate(([0.0], recalls, [1.0]))
    m_precisions = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(m_precisions) - 1, 0, -1):
        m_precisions[i - 1] = np.maximum(m_precisions[i - 1], m_precisions[i])

    indices = np.where(m_recalls[1:] != m_recalls[:-1])[0]
    ap = np.sum((m_recalls[indices + 1] - m_recalls[indices]) * m_precisions[indices + 1])

    # --- LƯU KẾT QUẢ ---
    base_dir = '/home/huy/Documents/RetinaNet_FreeAnchor_DIoU/Drone Thermal.v4i.voc'
    output_path = os.path.join(base_dir, 'recall_precision_csv/RetinaNet_anchorfree_DIoU')
    os.makedirs(output_path, exist_ok=True)

    # Lưu CSV
    pr_df = pd.DataFrame({'recall': recalls, 'precision': precisions})
    csv_file = os.path.join(output_path, 'recall_precision.csv')
    pr_df.to_csv(csv_file, index=False)

    # Vẽ biểu đồ
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, color='red', lw=2, label=f'mAP@0.5 = {ap * 100:.2f}%')
    plt.fill_between(recalls, precisions, alpha=0.2, color='red')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Tiled Inference)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    plt.legend(loc="lower left")

    plot_file = os.path.join(output_path, 'pr_curve.png')
    plt.savefig(plot_file, dpi=300)
    plt.close()

    print("--------------------------------------------------")
    print(f"Kết quả thực nghiệm trên hệ tọa độ gốc (Tiling):")
    print(f" - Tổng Ground Truth: {total_gt_boxes}")
    print(f" - Tổng Predictions: {len(all_predictions)}")
    print(f" - AP: {ap * 100:.2f}%")
    print(f" - Đã lưu biểu đồ tại: {plot_file}")
    print("--------------------------------------------------")


if __name__ == '__main__':
    evaluate_model()