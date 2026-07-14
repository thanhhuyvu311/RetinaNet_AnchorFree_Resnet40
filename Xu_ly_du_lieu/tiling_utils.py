"""
Tiling utilities — dùng chung cho preprocessing (main_step_123.py) và inference (predict.py).

Cung cấp các hàm:
  - generate_tile_coords : tính tọa độ các tile trên ảnh gốc
  - get_bboxes_for_tile  : lọc và chuyển bbox sang hệ tọa độ tile
  - crop_and_save_tile   : crop và lưu tile ra đĩa
  - apply_global_nms     : NMS trên toàn ảnh sau khi gộp boxes từ tất cả tiles
"""

import os
import numpy as np
from PIL import Image


def generate_tile_coords(img_w, img_h, tile_w, tile_h, overlap=0.2):
    """
    Tạo danh sách tọa độ (x_offset, y_offset, tile_w, tile_h) cho từng tile.

    Tiles cuối hàng / cuối cột được đẩy sát mép ảnh để đảm bảo toàn bộ
    ảnh đều được cover, dù step không chia hết.

    Args:
        img_w, img_h : kích thước ảnh gốc (pixel)
        tile_w, tile_h : kích thước mỗi tile (pixel)
        overlap : tỉ lệ chồng lấp giữa 2 tile liền kề (0.0 – 1.0)

    Returns:
        list of (x_offset, y_offset, tile_w, tile_h)
    """
    step_x = max(1, int(tile_w * (1 - overlap)))
    step_y = max(1, int(tile_h * (1 - overlap)))

    #---Quet vung bat dau cho cac tile---
    # --- Trục X ---
    xs = list(range(0, img_w - tile_w + 1, step_x))
    # Tile cuối chạm sát cạnh phải nếu chưa cover hết
    if not xs or xs[-1] + tile_w < img_w:
        xs.append(max(0, img_w - tile_w))
    xs = list(dict.fromkeys(xs))   # loại trùng, giữ thứ tự

    # --- Trục Y ---
    ys = list(range(0, img_h - tile_h + 1, step_y))
    # Tile cuối chạm sát cạnh dưới nếu chưa cover hết
    if not ys or ys[-1] + tile_h < img_h:
        ys.append(max(0, img_h - tile_h))
    ys = list(dict.fromkeys(ys))   # loại trùng, giữ thứ tự

    return [(x, y, tile_w, tile_h) for y in ys for x in xs]


def get_bboxes_for_tile(bboxes, class_ids, tile_x, tile_y, tile_w, tile_h,
                        min_visibility):
    """
    Lọc và chuyển đổi bbox từ hệ tọa độ ảnh gốc sang hệ tọa độ tile.

    Một bbox được giữ lại khi ≥ min_visibility diện tích của nó nằm trong tile.
    Phần bbox nằm ngoài tile bị clip tại biên tile.

    Args:
        bboxes     : list of [x_c, y_c, w, h] trong hệ ảnh gốc (pixel)
        class_ids  : list of int class id tương ứng
        tile_x, tile_y, tile_w, tile_h : vị trí và kích thước tile trong ảnh gốc
        min_visibility : tỉ lệ diện tích tối thiểu phải nằm trong tile

    Returns:
        (new_bboxes, new_class_ids) — bbox trong hệ tọa độ tile,
        format [x_c, y_c, w, h]
    """
    new_bboxes = []
    new_class_ids = []

    for bbox, cls_id in zip(bboxes, class_ids):
        x_c, y_c, w, h = bbox

        # Chuyển [x_c, y_c, w, h] → [xmin, ymin, xmax, ymax]
        xmin = x_c - w / 2.0
        ymin = y_c - h / 2.0
        xmax = x_c + w / 2.0
        ymax = y_c + h / 2.0

        # Tính phần giao nhau giữa bbox và tile
        inter_xmin = max(xmin, tile_x)
        inter_ymin = max(ymin, tile_y)
        inter_xmax = min(xmax, tile_x + tile_w)
        inter_ymax = min(ymax, tile_y + tile_h)

        # Bỏ qua nếu không có giao
        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            continue

        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        bbox_area = w * h

        if bbox_area <= 0:
            continue

        # Kiểm tra tỉ lệ hiển thị trong tile
        if inter_area / bbox_area < min_visibility:
            continue

        # Chuyển sang hệ tọa độ tile (gốc tại góc trên trái của tile)
        new_xmin = inter_xmin - tile_x
        new_ymin = inter_ymin - tile_y
        new_xmax = inter_xmax - tile_x
        new_ymax = inter_ymax - tile_y

        new_x_c = (new_xmin + new_xmax) / 2.0
        new_y_c = (new_ymin + new_ymax) / 2.0
        new_w   = new_xmax - new_xmin
        new_h   = new_ymax - new_ymin

        new_bboxes.append([new_x_c, new_y_c, new_w, new_h])
        new_class_ids.append(cls_id)

    return new_bboxes, new_class_ids


def crop_and_save_tile(image_path, tx, ty, tw, th, output_dir, tile_idx):
    """
    Crop một tile từ ảnh gốc và lưu ra đĩa (JPEG, quality=95).

    Args:
        image_path : đường dẫn ảnh gốc
        tx, ty     : offset (x, y) của tile trong ảnh gốc
        tw, th     : width, height của tile
        output_dir : thư mục lưu tile
        tile_idx   : chỉ số tile (dùng để đặt tên file, 3 chữ số)

    Returns:
        tile_path : đường dẫn file tile đã lưu
    """
    img = Image.open(image_path)
    tile = img.crop((tx, ty, tx + tw, ty + th))

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    tile_filename = f"{base_name}_tile_{tile_idx:03d}.jpg"
    tile_path = os.path.join(output_dir, tile_filename)
    tile.save(tile_path, "JPEG", quality=95)
    return tile_path


def apply_global_nms(boxes, scores, iou_threshold=0.45, score_threshold=0.05):
    """
    NMS (Non-Maximum Suppression) trên toàn bộ predictions sau khi gộp
    boxes từ tất cả tiles. Loại bỏ box trùng lặp ở vùng overlap.

    BẮT BUỘC phải gọi hàm này sau merge_tile_detections — nếu không,
    1 người ở vùng overlap sẽ bị detect 2 lần (2 box chồng nhau).

    Args:
        boxes          : list/array of [ymin, xmin, ymax, xmax] trong hệ ảnh gốc
        scores         : list/array of float confidence score tương ứng
        iou_threshold  : ngưỡng IoU để coi 2 box là trùng nhau
        score_threshold: loại bỏ box có score thấp hơn ngưỡng này

    Returns:
        (final_boxes, final_scores, num_det)
        final_boxes shape: (N, 4), format [ymin, xmin, ymax, xmax]
    """
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), 0

    boxes  = np.array(boxes,  dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    # Lọc theo score threshold
    keep_mask = scores >= score_threshold
    boxes  = boxes[keep_mask]
    scores = scores[keep_mask]

    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), 0

    # Sắp xếp theo score giảm dần
    order  = np.argsort(scores)[::-1]
    boxes  = boxes[order]
    scores = scores[order]

    y1, x1, y2, x2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (y2 - y1) * (x2 - x1)

    kept        = []
    suppressed  = np.zeros(len(boxes), dtype=bool)

    for i in range(len(boxes)):
        if suppressed[i]:
            continue
        kept.append(i)

        # Tính IoU của box i với tất cả box phía sau (vectorized)
        inter_y1 = np.maximum(y1[i], y1[i + 1:])
        inter_x1 = np.maximum(x1[i], x1[i + 1:])
        inter_y2 = np.minimum(y2[i], y2[i + 1:])
        inter_x2 = np.minimum(x2[i], x2[i + 1:])

        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_area = inter_h * inter_w

        iou = inter_area / (areas[i] + areas[i + 1:] - inter_area + 1e-8)
        suppressed[i + 1:][iou > iou_threshold] = True

    final_boxes  = boxes[kept]
    final_scores = scores[kept]
    return final_boxes, final_scores, len(kept)
