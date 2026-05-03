#!/usr/bin/env python3
"""YOLO v3 object detection class"""

import numpy as np
import tensorflow.keras as K


class Yolo:
    """Uses the YOLO v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor

        Parameters:
        - model_path: path to Darknet Keras model
        - classes_path: path to class names file
        - class_t: box score threshold for filtering
        - nms_t: IOU threshold for non-max suppression
        - anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
        """

        # Load the Keras model
        self.model = K.models.load_model(model_path)

        # Load class names
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        # Set thresholds
        self.class_t = class_t
        self.nms_t = nms_t

        # Set anchor boxes
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Uses the YOLO v3 algorithm to perform object detection"""

        boxes = []
        box_confidences = []
        box_class_probs = []

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        image_h, image_w = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            box_conf = output[..., 4:5]
            class_probs = output[..., 5:]

            # Grid offsets
            c_x = np.arange(grid_w).reshape(1, grid_w, 1)
            c_x = np.tile(c_x, (grid_h, 1, anchor_boxes))

            c_y = np.arange(grid_h).reshape(grid_h, 1, 1)
            c_y = np.tile(c_y, (1, grid_w, anchor_boxes))

            # Normalized box center
            b_x = (1 / (1 + np.exp(-t_x)) + c_x) / grid_w
            b_y = (1 / (1 + np.exp(-t_y)) + c_y) / grid_h

            # Normalize anchor dimensions by model input size
            anchor_w = self.anchors[i, :, 0].reshape(1, 1, anchor_boxes)
            anchor_h = self.anchors[i, :, 1].reshape(1, 1, anchor_boxes)

            b_w = (anchor_w * np.exp(t_w)) / input_w  # ← fix
            b_h = (anchor_h * np.exp(t_h)) / input_h  # ← fix

            # Convert normalized coords to image pixel coords
            x1 = (b_x - b_w / 2) * image_w
            y1 = (b_y - b_h / 2) * image_h
            x2 = (b_x + b_w / 2) * image_w
            y2 = (b_y + b_h / 2) * image_h

            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))
            box_confidences.append(1 / (1 + np.exp(-box_conf)))
            box_class_probs.append(1 / (1 + np.exp(-class_probs)))

        return boxes, box_confidences, box_class_probs
