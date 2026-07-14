import os
os.environ['TF_DATA_AUTOTUNE_RAM_BUDGET'] = '500000000'

import tensorflow as tf
from tensorflow.keras import mixed_precision
from Anchor_box import Anchor_box
from Label_encode import LabelEncoder
from Xu_ly_du_lieu import preprocessing_data_before_training
from RetinaNet import RetinaNet, RetinaNetLoss
from resnet_40 import resnet_40_backbone
import pandas as pd
import ast


def pack_targets(img, bdb, cls):
    # Lấy target_boxes (4 cột) và target_classes (1 cột) từ encoder
    # Mã hóa nhãn (Bounding box và Class) bằng bộ Anchor Box (đã được tối ưu nhỏ lại)
    _, target_boxes, target_classes = label_encoder.encode_batch(img, bdb, cls, all_anchor)

    # Ép kiểu target_classes về float32 để đồng nhất với target_boxes
    target_classes = tf.cast(target_classes, tf.float32)

    # Mở rộng chiều của target_classes để nó có dạng (Batch, Num_Anchor, 1)
    target_classes = tf.expand_dims(target_classes, axis=-1)

    # Hàn 2 cái lại thành 1 cục y_true duy nhất có 5 cột (Batch, Num_Anchor, 5)
    y_true = tf.concat([target_boxes, target_classes], axis=-1)

    # Trả về ĐÚNG 2 món: Ảnh (Input) và Nhãn gộp (Target) cho Keras
    return img, y_true


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    base_dir = '/home/huy/Documents/RetinaNet_FreeAnchor_DIoU_resnet40/Drone Thermal.v4i.voc'

    target_size = 512  # [TILING - ĐỔI] 640 → 512: tile 640×512 resize về 512×512
    batch_size = 8  # [TILING - ĐỔI] 4 → 8: tile nhỏ hơn → tăng batch_size được

    # ---KHOI TAO ANCHOR BOX ---#
    anchor_gene = Anchor_box()
    # tao anchor
    all_anchor = anchor_gene.get_anchors(img_h=target_size, img_w=target_size)
    # khoi tao label encoder
    label_encoder = LabelEncoder()

    # --- XU LY DATASET VA GAN NHAN ---#
    # Đọc đường dẫn dữ liệu CSV (Dữ liệu đã được tiling/cắt nhỏ)
    train_csv_path = '/home/huy/Documents/RetinaNet_FreeAnchor_DIoU_resnet40/Drone Thermal.v4i.voc/csv_file/train_data_6_7_2026.csv'
    val_csv_path = '/home/huy/Documents/RetinaNet_FreeAnchor_DIoU_resnet40/Drone Thermal.v4i.voc/csv_file/valid_data_6_7_2026.csv'
    raw_train_data = preprocessing_data_before_training.change_string2number(train_csv_path)
    raw_val_data = preprocessing_data_before_training.change_string2number(val_csv_path)
    train_dataset = preprocessing_data_before_training.create_dataset_from_dataframe(raw_train_data).shuffle(
        buffer_size=1000)
    val_dataset = preprocessing_data_before_training.create_dataset_from_dataframe(raw_val_data)

    # Đọc ảnh, resize và thêm Data Augmentation chuyên cho ảnh nhiệt (Thermal)
    train_dataset = train_dataset.map(preprocessing_data_before_training.read_img_and_label,
                                      num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.map(
        lambda img, bdb, cls: preprocessing_data_before_training.resize_and_pad_img(img, bdb, cls, target_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    # [TỐI ƯU DRONE] Áp dụng Data Augmentation (Lật ngang, độ sáng, tương phản)
    # Giúp mô hình bắt được người ở mọi hướng và trong điều kiện nhiệt độ môi trường khác nhau
    train_dataset = train_dataset.map(
        preprocessing_data_before_training.data_augment,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    val_dataset = val_dataset.map(preprocessing_data_before_training.read_img_and_label,
                                  num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(
        lambda img, bdb, cls: preprocessing_data_before_training.resize_and_pad_img(img, bdb, cls, target_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    train_dataset = train_dataset.padded_batch(
        batch_size=batch_size,
        padding_values=(0.0, 1e-8, -2),
        drop_remainder=True  # cat bo nhung anh le khong cung 1 batch
    )

    train_dataset = train_dataset.map(
        pack_targets,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    val_dataset = val_dataset.padded_batch(
        batch_size=batch_size,
        padding_values=(0.0, 1e-8, -2),
        drop_remainder=True  # cat bo nhung anh le khong cung 1 batch
    )

    val_dataset = val_dataset.map(
        pack_targets,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # [FIX RAM]: Chốt cứng buffer_size=2 thay vì để AUTOTUNE bòn rút RAM
    train_dataset = train_dataset.prefetch(buffer_size=2)
    val_dataset = val_dataset.prefetch(buffer_size=2)

    # ------------------------------#
    #                              #
    #      Khoi tao mo hinh        #
    # ------------------------------#

    train_df = pd.read_csv(train_csv_path)
    all_labels = []
    for cid in train_df['class_id'].apply(ast.literal_eval):
        all_labels.extend(cid)

    num_class = len(set(all_labels))
    num_classes = num_class

    resnet40_backbone = resnet_40_backbone()
    model = RetinaNet(num_classes=num_classes, backbone=resnet40_backbone)

    # [TỐI ƯU LOSS & GRADIENT] Bổ sung clipnorm=10.0 để tránh bùng nổ Gradient khi dùng Loss CIoU/DIoU
    # Learning rate đặt ở mức nhỏ (1e-4) do bài toán fine-tuning trên vật thể nhỏ thường rất nhạy cảm
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=10.0)
    loss_fn = RetinaNetLoss(anchors=all_anchor, num_classes=num_classes)

    # 3. Compile model
    model.compile(optimizer=optimizer, loss=loss_fn)

    # 4. Thiết lập Callbacks để lưu trọng số và tự dừng
    weight_folder = os.path.join(base_dir, 'weight_store')
    os.makedirs(weight_folder, exist_ok=True)

    callbacks_list = [
        # Tự động ngắt hệ thống nếu Loss văng ra NaN (Bảo vệ tiến trình train)
        tf.keras.callbacks.TerminateOnNaN(),

        # Giảm Learning Rate khi mô hình có dấu hiệu chững lại (Plateau)
        # Giúp mô hình hội tụ tốt hơn ở các epoch cuối
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", 
            factor=0.5, 
            patience=3, 
            min_lr=1e-7, 
            verbose=1 
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=weight_folder + "/RetinaNet_anchorfree_DIoU_resnet40.weights.h5",
            # [TILING - ĐỔI] không ghi đè model cũ
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            verbose=1,
            restore_best_weights=True
        )
    ]

    EPOCHS = 200

    hist = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks_list,
    )
    hist_df = pd.DataFrame(hist.history)

    model_store_folder = os.path.join(base_dir, 'model_store')
    hist_store_folder = os.path.join(base_dir, 'hist_store')
    os.makedirs(hist_store_folder, exist_ok=True)
    hist_df.to_csv(hist_store_folder + '/RetinaNet_anchorfree_DIoU_resnet40.csv')