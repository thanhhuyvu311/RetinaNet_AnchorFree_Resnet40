import keras.layers
import tensorflow as tf
from resnet_40 import resnet_40_backbone


class FeaturePyramid(keras.layers.Layer):
    """
    Xay dung dac trung kim tu thap tu cac ban do dac trung cua backbone.
    Su dung ResNet-40 (bo P5), chi xuat ra P3 va P4.
    """

    def __init__(self, backbone=None, **kwargs):
        super().__init__(name='FeaturePyramid', **kwargs)
        self.backbone = backbone if backbone else resnet_40_backbone()

        self.conv_c3_1x1 = keras.layers.Conv2D(256, 1, 1, padding='same', name='conv_c3_1x1')
        self.conv_c4_1x1 = keras.layers.Conv2D(256, 1, 1, padding='same', name='conv_c4_1x1')

        self.conv_c3_3x3 = keras.layers.Conv2D(256, 3, 1, padding='same', name='conv_c3_3x3')
        self.conv_c4_3x3 = keras.layers.Conv2D(256, 3, 1, padding='same', name='conv_c4_3x3')

        self.upsample_2x = keras.layers.UpSampling2D(2)

    def call(self, images, training=False):
        c3_output, c4_output = self.backbone(images, training=training)

        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)

        # Top-down pathway: p3 nhan them thong tin tu p4
        p3_output = p3_output + self.upsample_2x(p4_output)

        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)

        return p3_output, p4_output