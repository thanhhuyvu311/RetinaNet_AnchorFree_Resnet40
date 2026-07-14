import tensorflow as tf

class Anchor_box:
    """
    Anchor-free point generator for feature map locations.
    Generates [cx, cy, stride, stride] for each location.
    """
    def __init__(self):
        self._num_anchors = 1
        self._strides = [2 ** i for i in range(3, 5)]

    def _get_anchors(self, feature_h, feature_w, level):
        ry = tf.range(feature_w, dtype=tf.float32) + 0.5 
        rx = tf.range(feature_h, dtype=tf.float32) + 0.5

        X, Y = tf.meshgrid(ry, rx) 

        centers = tf.stack([X, Y], axis=-1) * self._strides[level - 3] 
        centers = tf.reshape(centers, [-1, 2])
        
        stride_tensor = tf.fill([tf.shape(centers)[0], 2], tf.cast(self._strides[level-3], tf.float32))
        anchors = tf.concat([centers, stride_tensor], axis=-1)
        
        return anchors

    def get_anchors(self, img_h, img_w):
        anchors = [
            self._get_anchors(
                tf.math.ceil(img_h/2 ** i),
                tf.math.ceil(img_w/2 ** i),
                i,
            )
            for i in range(3, 5)
        ]
        return tf.concat(anchors, axis=0)
