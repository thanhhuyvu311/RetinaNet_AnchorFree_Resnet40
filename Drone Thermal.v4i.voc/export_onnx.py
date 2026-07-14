import os

# Buoc 1: Ep TensorFlow chay tren CPU de tranh loi 137 (OOM) tren GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import tf2onnx
import numpy as np
import sys
sys.path.append('/home/huy/Documents/RetinaNet_FreeAnchor_DIoU')
from RetinaNet import RetinaNet
from resnet_40 import resnet_40_backbone

# --- CAU HINH DUONG DAN ---
# Ong thay doi cac duong dan nay cho dung voi may cua minh
TARGET_SIZE = 512
NUM_CLASSES = 1
WEIGHT_PATH = '/home/huy/Documents/RetinaNet_FreeAnchor_DIoU/Drone Thermal.v4i.voc/weight_store/RetinaNet_L1_7_7_2026_17_23.weights.h5'
OUTPUT_ONNX = 'RetinaNet_L1_7_7_2026_17_23.onnx'


def export_to_onnx():
    print("--- Dang khoi tao mo hinh RetinaNet-ResNet50 ---")

    try:
        # 1. Khoi tao lai cau truc mang (Backbone + Head)
        backbone = resnet_40_backbone()
        #model = RetinaNet(num_classes=NUM_CLASSES, backbone=backbone) dung cho free anchor
        model = RetinaNet(num_classes=NUM_CLASSES, backbone=backbone, num_anchors=9)

        # 2. Build graph bang cach chay thu mot dummy tensor
        # Dung float32 vi ONNX thuong yeu cau kieu du lieu chuan nay
        dummy_input = tf.zeros((1, TARGET_SIZE, TARGET_SIZE, 3), dtype=tf.float32)
        _ = model(dummy_input)

        # 3. Load trong so (.h5)
        print(f"--- Dang load weights tu: {WEIGHT_PATH} ---")
        model.load_weights(WEIGHT_PATH)
        print("--- Load weights thanh cong! ---")

        # 4. Dinh nghia Input Signature (Batch size = 1, co dinh 512x512)
        spec = (tf.TensorSpec((1, TARGET_SIZE, TARGET_SIZE, 3), tf.float32, name="input"),)

        # 5. Thuc hien chuyen doi bang tf2onnx
        print(f"--- Dang bat dau chuyen doi sang ONNX (Vui long doi...) ---")
        model_proto, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=spec,
            opset=13,  # Phien ban opset on dinh nhat cho Jetson TensorRT
            output_path=OUTPUT_ONNX
        )

        print(f"--- CHUC MUNG! Da luu file tai: {OUTPUT_ONNX} ---")
        print("--- Gio ong co the mang file nay sang Jetson de build Engine ---")

    except Exception as e:
        print(f"Co loi xay ra: {e}")


if __name__ == '__main__':
    export_to_onnx()