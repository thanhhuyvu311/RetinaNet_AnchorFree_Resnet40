import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast


def analyze_object_sizes(csv_path, target_size):
    df = pd.read_csv(csv_path)
    all_sides = []
    all_ratios = []
    for index, row in df.iterrows():
        # 1. Lay thong tin
        bboxes = ast.literal_eval(row['bbox'])


        # [TILING - ĐỔI] bbox trong tiled CSV đã ở hệ tile (640×512), không phải ảnh gốc 1280×1024
        # → orig_w, orig_h phải là kích thước TILE, không phải ảnh gốc
        orig_w, orig_h = 720, 576   # kích thước tile (đổi lại nếu dùng tile size khác)
        ratio = target_size / max(orig_w, orig_h)

        for box in bboxes:
            # box dang [x, y, w, h]
            w_orig, h_orig = box[2], box[3]

            aspect_ratio = w_orig / h_orig
            all_ratios.append(aspect_ratio)
            # 2. Quy doi ve kich thuoc sau khi resize
            w_res = w_orig * ratio
            h_res = h_orig * ratio

            # 3. Tinh canh tuong duong
            side = np.sqrt(w_res * h_res)
            all_sides.append(side)

    # 4. Ve Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_sides, bins=50, color='skyblue', edgecolor='black')
    plt.title(f'Phân bổ kích thước vật thể (Side Length) sau khi Resize về {target_size}')
    plt.xlabel('Cạnh hình vuông tương đương (pixel)')
    plt.ylabel('Số lượng vật thể')
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(all_ratios, bins=50, color='skyblue', edgecolor='black')
    plt.title(f'Phân bổ ratios sau khi Resize về {target_size}')
    plt.grid(True, alpha=0.3)
    plt.show()

    # 5. Goi y thong so
    print("===== KICH THUOC VAT THE (sau resize) =====")
    print(f"Kich thuoc nho nhat: {min(all_sides):.2f}")
    print(f"Kich thuoc lon nhat: {max(all_sides):.2f}")
    print(f"Kich thuoc trung binh: {np.mean(all_sides):.2f}")
    print(f"Phan vi 25% (25th percentile): {np.percentile(all_sides, 25):.2f}")
    print(f"Phan vi 75% (75th percentile): {np.percentile(all_sides, 75):.2f}")

    print("\n===== TI LE KHUNG HINH (aspect ratio = w/h) =====")
    print(f"Ratio nho nhat: {min(all_ratios):.2f}")
    print(f"Ratio lon nhat: {max(all_ratios):.2f}")
    print(f"Ratio trung binh: {np.mean(all_ratios):.2f}")
    print(f"Phan vi 10% (10th percentile): {np.percentile(all_ratios, 10):.2f}")
    print(f"Phan vi 25% (25th percentile): {np.percentile(all_ratios, 25):.2f}")
    print(f"Phan vi 75% (75th percentile): {np.percentile(all_ratios, 75):.2f}")
    print(f"Phan vi 90% (90th percentile): {np.percentile(all_ratios, 90):.2f}")


# [TILING - ĐỔI] chạy trên CSV tile mới, target_size = 640 (hoặc 512 tùy bạn chọn)
analyze_object_sizes(
    '/home/huy/Documents/de_tai_tot_nghiep/Drone Thermal.v4i.voc/csv_file/data_information_tiled_2img.csv',
    target_size=512
)