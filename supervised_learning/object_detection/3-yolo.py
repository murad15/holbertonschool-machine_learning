#!/usr/bin/env python3
"""YOLO object detection"""


import numpy as np


class Yolo:
    """Uses the YOLO v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        model_path: path to where a Darknet Keras model is stored
        classes_path: path to where the list of class names used for the model is stored
        class_t: box score threshold for initial filtering
        nms_t: IOU threshold for non-max suppression
        anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
        """
        from tensorflow import keras

        self.model = keras.models.load_model(model_path)
        self.class_names = self.load_classes(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def load_classes(classes_path):
        """
        Loads class names from a file
        """
        with open(classes_path, "r") as f:
            return [line.strip() for line in f.readlines()]

    @staticmethod
    def iou(box1, box2):
        """
        Calculates Intersection over Union between two boxes

        box format:
        [x1, y1, x2, y2]
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0

        return inter_area / union_area

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies Non-max Suppression to filtered bounding boxes.

        filtered_boxes: numpy.ndarray of shape (?, 4)
        box_classes: numpy.ndarray of shape (?,)
        box_scores: numpy.ndarray of shape (?,)

        Returns:
            box_predictions: numpy.ndarray of shape (?, 4)
            predicted_box_classes: numpy.ndarray of shape (?,)
            predicted_box_scores: numpy.ndarray of shape (?,)
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            cls_indices = np.where(box_classes == cls)[0]

            cls_boxes = filtered_boxes[cls_indices]
            cls_scores = box_scores[cls_indices]

            sorted_indices = np.argsort(cls_scores)[::-1]

            while sorted_indices.size > 0:
                best_index = sorted_indices[0]

                box_predictions.append(cls_boxes[best_index])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(cls_scores[best_index])

                remaining_indices = sorted_indices[1:]

                keep_indices = []

                for idx in remaining_indices:
                    iou_score = self.iou(cls_boxes[best_index], cls_boxes[idx])

                    if iou_score <= self.nms_t:
                        keep_indices.append(idx)

                sorted_indices = np.array(keep_indices)

        return (
            np.array(box_predictions),
            np.array(predicted_box_classes),
            np.array(predicted_box_scores),
        )
