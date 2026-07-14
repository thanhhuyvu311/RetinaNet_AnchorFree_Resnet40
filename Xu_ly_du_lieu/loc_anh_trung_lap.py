import cv2
import numpy as np
import pandas as pd


def calculate_mse(imageA, imageB):
    # Tính sai số toàn phương trung bình (MSE) giữa 2 ảnh
    # MSE càng nhỏ thì 2 ảnh càng giống nhau. Bằng 0 là giống hệt nhau.
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def clean_duplicate_dataset(csv_path, output_path, mse_threshold=5.0):
    print("Đang tải file CSV...")
    df = pd.read_csv(csv_path)

    # Đảm bảo dữ liệu được sắp xếp theo tên file để các ảnh giống nhau nằm kề nhau
    df = df.sort_values(by='path_img').reset_index(drop=True)

    cleaned_rows = []
    last_image = None

    print(f"Bắt đầu lọc {len(df)} bức ảnh...")

    for index, row in df.iterrows():
        img_path = row['path_img']
        # Đọc ảnh dưới dạng Grayscale (ảnh nhiệt dùng grayscale là đủ so sánh)
        current_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if current_image is None:
            continue  # Bỏ qua nếu đường dẫn ảnh bị lỗi

        if last_image is None:
            # Lưu bức ảnh đầu tiên vào danh sách sạch
            cleaned_rows.append(row)
            last_image = current_image
            continue

        # Kiểm tra xem ảnh hiện tại và ảnh kích thước có bằng nhau không
        if current_image.shape == last_image.shape:
            # Tính độ lệch MSE
            err = calculate_mse(current_image, last_image)
            print(f"Độ lệch MSE giữa 2 ảnh liên tiếp là: {err}")
            # Nếu lệch nhiều hơn ngưỡng cho phép -> Bức ảnh này có sự thay đổi (người di chuyển)
            if err > mse_threshold:
                cleaned_rows.append(row)
                last_image = current_image  # Cập nhật lại cột mốc so sánh
        else:
            # Kích thước khác nhau thì chắc chắn là ảnh khác, giữ lại
            cleaned_rows.append(row)
            last_image = current_image

    # Xuất ra file CSV mới
    cleaned_df = pd.DataFrame(cleaned_rows)
    cleaned_df.to_csv(output_path, index=False)

    print("========================================")
    print(f"Tổng số ảnh ban đầu: {len(df)}")
    print(f"Tổng số ảnh giữ lại: {len(cleaned_df)}")
    print(f"Đã loại bỏ: {len(df) - len(cleaned_df)} ảnh rác/trùng lặp!")
    print(f"File CSV sạch được lưu tại: {output_path}")


if __name__ == '__main__':
    # Đường dẫn file CSV gốc của ông
    input_csv = '/home/huy/Documents/de_tai_tot_nghiep/Drone Thermal.v4i.voc/csv_file/train_data_6_7_2026.csv'

    # File mới sau khi dọn dẹp
    output_csv = '/home/huy/Documents/de_tai_tot_nghiep/Drone Thermal.v4i.voc/csv_file/train_data_6_7_2026_cleaned.csv'

    # Chạy hàm lọc
    clean_duplicate_dataset(input_csv, output_csv, mse_threshold=5.0)