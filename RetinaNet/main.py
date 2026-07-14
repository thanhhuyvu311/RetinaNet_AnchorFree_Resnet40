import tensorflow as tf
from FPN import FeaturePyramid
from resnet_40 import head_of_model
import numpy as np
import keras.layers

class RetinaNet(keras.Model):
    """A subclassed Keras model implementing the RetinaNet architecture.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50 only.
    """

    def __init__(self, num_classes, backbone=None, num_anchors=1, **kwargs):
        super().__init__(name="RetinaNet", **kwargs)
        self.fpn = FeaturePyramid(backbone)
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.cls_head = head_of_model(self.num_anchors * num_classes, prior_probability)  #phan du doan nhan
        self.box_head = head_of_model(self.num_anchors * 4, "zeros") # phan du doan box

    def call(self, image, training=False):
        features = self.fpn(image, training=training)
        N = tf.shape(image)[0]
        cls_outputs = []
        box_outputs = []
        for feature in features:
            box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
            cls_outputs.append(
                tf.reshape(self.cls_head(feature), [N, -1, self.num_classes])
            )
        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)
        return tf.concat([box_outputs, cls_outputs], axis=-1)


#boxloss retina net
class RetinaNetBoxLoss(tf.losses.Loss):
    """Implements DIoU loss"""

    def __init__(self, anchors):
        super().__init__(
            reduction="none", name="RetinaNetBoxLoss"
        )
        self.anchors = anchors
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _decode_box_predictions(self, encoded_boxes):
        # encoded_boxes: [batch, num_anchors, 4]
        # self.anchors: [num_anchors, 4]
        box_variance = tf.cast(self._box_variance, encoded_boxes.dtype)
        encoded_boxes = encoded_boxes * box_variance

        anchors = tf.cast(self.anchors, encoded_boxes.dtype)
        anchors = tf.expand_dims(anchors, axis=0) # [1, num_anchors, 4]
        
        anchor_cx_cy = anchors[..., :2]
        anchor_stride = anchors[..., 2:]

        # Decode centers
        decoded_cx_cy = encoded_boxes[..., :2] * anchor_stride + anchor_cx_cy
        # Decode width and height
        decoded_w_h = tf.math.exp(encoded_boxes[..., 2:]) * anchor_stride

        # Convert back to x1, y1, x2, y2 format for DIoU calculation
        decoded_x1_y1 = decoded_cx_cy - decoded_w_h / 2
        decoded_x2_y2 = decoded_cx_cy + decoded_w_h / 2
        
        return tf.concat([decoded_x1_y1, decoded_x2_y2], axis=-1)

    def call(self, y_true, y_pred):
        y_true_decoded = self._decode_box_predictions(y_true)
        y_pred_decoded = self._decode_box_predictions(y_pred)

        b1_x1, b1_y1, b1_x2, b1_y2 = tf.split(y_true_decoded, 4, axis=-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = tf.split(y_pred_decoded, 4, axis=-1)
        
        # Intersection
        inter_x1 = tf.maximum(b1_x1, b2_x1)
        inter_y1 = tf.maximum(b1_y1, b2_y1)
        inter_x2 = tf.minimum(b1_x2, b2_x2)
        inter_y2 = tf.minimum(b1_y2, b2_y2)
        
        inter_w = tf.maximum(0.0, inter_x2 - inter_x1)
        inter_h = tf.maximum(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        
        # Union
        b1_w = tf.maximum(0.0, b1_x2 - b1_x1)
        b1_h = tf.maximum(0.0, b1_y2 - b1_y1)
        b2_w = tf.maximum(0.0, b2_x2 - b2_x1)
        b2_h = tf.maximum(0.0, b2_y2 - b2_y1)
        
        b1_area = b1_w * b1_h
        b2_area = b2_w * b2_h
        union_area = tf.maximum(b1_area + b2_area - inter_area, 1e-7)
        
        iou = inter_area / union_area
        
        # Enclosing box
        enclose_x1 = tf.minimum(b1_x1, b2_x1)
        enclose_y1 = tf.minimum(b1_y1, b2_y1)
        enclose_x2 = tf.maximum(b1_x2, b2_x2)
        enclose_y2 = tf.maximum(b1_y2, b2_y2)
        
        enclose_w = tf.maximum(0.0, enclose_x2 - enclose_x1)
        enclose_h = tf.maximum(0.0, enclose_y2 - enclose_y1)
        
        # Diagonal length squared of enclosing box
        c_squared = tf.square(enclose_w) + tf.square(enclose_h) + 1e-7
        
        # Distance squared between centers
        b1_cx = b1_x1 + b1_w / 2
        b1_cy = b1_y1 + b1_h / 2
        b2_cx = b2_x1 + b2_w / 2
        b2_cy = b2_y1 + b2_h / 2
        
        rho_squared = tf.square(b2_cx - b1_cx) + tf.square(b2_cy - b1_cy)
        
        diou = iou - (rho_squared / c_squared)
        loss = 1.0 - diou
        
        return tf.squeeze(loss, axis=-1)
class RetinaNetClassificationLoss(tf.losses.Loss):
    """Implements Focal loss"""

    def __init__(self, alpha, gamma):
        super().__init__(
            reduction="none", name="RetinaNetClassificationLoss"
        )
        self._alpha = alpha
        self._gamma = gamma

    def call(self, y_true, y_pred):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        probs = tf.nn.sigmoid(y_pred)
        alpha = tf.where(tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha))
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)
        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

class RetinaNetLoss(tf.losses.Loss):
    """Wrapper to combine both the losses"""
    #base alpha 0.25, gamma 2.0
    """
    gamma la tham so tap trung, neu mo hinh doan nham vat the khac la nguoi thi tang chi so nay len
    alpha la tham so can bang, neu mo hinh bat sot qua nhieu (recall thap) thi tang chi so nay len de uu tien nguoi hon
    """
    def __init__(self, anchors, num_classes=80, alpha=0.25, gamma=2.0):
        super().__init__(reduction="sum_over_batch_size", name="RetinaNetLoss")
        self._clf_loss = RetinaNetClassificationLoss(alpha, gamma)
        self._box_loss = RetinaNetBoxLoss(anchors)
        self._num_classes = num_classes

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        box_labels = y_true[:, :, :4]
        box_predictions = y_pred[:, :, :4]
        cls_labels = tf.one_hot(
            tf.cast(y_true[:, :, 4], dtype=tf.int32),
            depth=self._num_classes,
            dtype=tf.float32,
        )
        cls_predictions = y_pred[:, :, 4:]

        positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32)
        ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=tf.float32)

        clf_loss = self._clf_loss(cls_labels, cls_predictions)
        box_loss = self._box_loss(box_labels, box_predictions)

        # Cắt bỏ những vùng không cần thiết (Nền và Ignore)
        clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)

        # THÊM NORMALIZER TẠI ĐÂY: Đếm số lượng Positive Anchor trên mỗi ảnh
        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        normalizer = tf.maximum(1.0, normalizer)  # Ép tối thiểu là 1 để tránh lỗi chia cho 0

        # Chia tổng Loss cho Normalizer để ổn định Gradient
        clf_loss = tf.reduce_sum(clf_loss, axis=-1) / normalizer
        box_loss = tf.reduce_sum(box_loss, axis=-1) / normalizer

        return clf_loss + box_loss