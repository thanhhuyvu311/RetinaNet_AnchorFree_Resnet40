import tensorflow as tf

class LabelEncoder:
    def __init__(self):
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _compute_box_target(self, points, matched_gt_boxes):
        # points: [cx, cy, stride, stride]
        # matched_gt_boxes: [gt_cx, gt_cy, gt_w, gt_h]
        box_target_xy = (matched_gt_boxes[:, :2] - points[:, :2]) / points[:, 2:]

        safe_gt_wh = tf.math.maximum(matched_gt_boxes[:, 2:], 1e-7)
        box_target_wh = tf.math.log(safe_gt_wh / points[:, 2:])

        box_target = tf.concat([box_target_xy, box_target_wh], axis=-1)
        return box_target / self._box_variance

    def _encode_sample(self, gt_boxes, gt_classes, points):
        if tf.shape(gt_boxes)[0] == 0:
            return tf.zeros_like(points), tf.fill([tf.shape(points)[0]], -1.0)
        
        gt_centers = gt_boxes[:, :2]
        point_centers = points[:, :2]
        
        pc_x = tf.expand_dims(point_centers[:, 0], 1)
        pc_y = tf.expand_dims(point_centers[:, 1], 1)
        
        gc_x = tf.expand_dims(gt_boxes[:, 0], 0)
        gc_y = tf.expand_dims(gt_boxes[:, 1], 0)
        gw = tf.expand_dims(gt_boxes[:, 2], 0)
        gh = tf.expand_dims(gt_boxes[:, 3], 0)
        
        inside_x = tf.logical_and(pc_x >= gc_x - gw/2.0, pc_x <= gc_x + gw/2.0)
        inside_y = tf.logical_and(pc_y >= gc_y - gh/2.0, pc_y <= gc_y + gh/2.0)
        is_inside = tf.logical_and(inside_x, inside_y)
        
        diff = tf.expand_dims(point_centers, 1) - tf.expand_dims(gt_centers, 0)
        distances = tf.reduce_sum(tf.square(diff), axis=-1)
        closest_point_idx = tf.argmin(distances, axis=0)
        
        is_closest = tf.equal(
            tf.expand_dims(tf.range(tf.shape(points)[0], dtype=tf.int64), 1),
            tf.expand_dims(closest_point_idx, 0)
        )
        
        is_assigned = tf.logical_or(is_inside, is_closest)
        
        gt_area = gw * gh
        area_matrix = tf.where(is_assigned, gt_area, tf.fill(tf.shape(is_assigned), float('inf')))
        
        min_area_idx = tf.argmin(area_matrix, axis=1)
        min_area = tf.reduce_min(area_matrix, axis=1)
        
        positive_mask = tf.less(min_area, float('inf'))
        
        matched_gt_boxes = tf.gather(gt_boxes, min_area_idx)
        matched_gt_classes = tf.gather(gt_classes, min_area_idx)
        
        target_classes = tf.cast(tf.fill([tf.shape(points)[0]], -1.0), tf.float32)
        target_classes = tf.where(positive_mask, tf.cast(matched_gt_classes, tf.float32), target_classes)
        
        target_boxes = self._compute_box_target(points, matched_gt_boxes)
        
        return target_boxes, target_classes

    def encode_batch(self, batch_images, gt_boxes, gt_classes, anchor_boxes):
        target_boxes, target_classes = tf.map_fn(
            fn=lambda x: self._encode_sample(x[0], x[1], anchor_boxes),
            elems=(gt_boxes, gt_classes),
            fn_output_signature=(tf.float32, tf.float32)
        )
        return batch_images, target_boxes, target_classes