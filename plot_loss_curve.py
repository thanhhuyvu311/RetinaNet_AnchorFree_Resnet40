import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_loss(csv_path, save_path=None):
    if not os.path.exists(csv_path):
        print(f"Lỗi: Không tìm thấy file tại {csv_path}")
        return

    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(csv_path)

    # Kiểm tra xem có cột loss và val_loss hay không
    if 'loss' not in df.columns or 'val_loss' not in df.columns:
        print("Lỗi: File CSV không chứa cột 'loss' hoặc 'val_loss'.")
        return

    loss = df['loss']
    val_loss = df['val_loss']
    epochs = range(1, len(loss) + 1)

    # Khởi tạo biểu đồ
    plt.figure(figsize=(10, 6))
    
    # Vẽ đường loss
    plt.plot(epochs, loss, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'r--', label='Validation Loss', linewidth=2)

    # Thêm tiêu đề và nhãn
    plt.title('Biểu đồ biểu diễn Training và Validation Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)

    # Ghi chú giá trị ở Epoch cuối cùng
    last_epoch = epochs[-1]
    last_loss = loss.iloc[-1]
    last_val_loss = val_loss.iloc[-1]

    plt.annotate(f'Train: {last_loss:.4f}', 
                 xy=(last_epoch, last_loss), xytext=(5, 10),
                 textcoords="offset points", color='blue', fontsize=10, fontweight='bold')
                 
    plt.annotate(f'Val: {last_val_loss:.4f}', 
                 xy=(last_epoch, last_val_loss), xytext=(5, -15),
                 textcoords="offset points", color='red', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Lưu biểu đồ thành file ảnh (nếu được yêu cầu)
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Đã lưu biểu đồ tại: {save_path}")

    # Hiển thị biểu đồ
    plt.show()

if __name__ == '__main__':
    # Đường dẫn tới file CSV chứa lịch sử huấn luyện
    CSV_PATH = '/home/huy/Documents/RetinaNet_FreeAnchor_DIoU/Drone Thermal.v4i.voc/hist_store/RetinaNet_anchorfree_DIoU.csv'
    
    # Nơi bạn muốn lưu ảnh biểu đồ (bạn có thể đổi tên tùy ý)
    SAVE_IMG_PATH = '/home/huy/Documents/RetinaNet_FreeAnchor_DIoU/Drone Thermal.v4i.voc/hist_store/loss_curve_RetinaNet.png'
    
    print("Đang tiến hành vẽ biểu đồ...")
    plot_loss(CSV_PATH, save_path=SAVE_IMG_PATH)
