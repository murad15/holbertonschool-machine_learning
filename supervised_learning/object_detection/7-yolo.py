#!/usr/bin/env python3
"""YOLO object detection"""

import os
import cv2
import numpy as np


class Yolo:
    """Uses the YOLO v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initializes the YOLO object detector"""
        from tensorflow import keras

        self.model = keras.models.load_model(model_path)
        self.class_names = self.load_classes(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def load_classes(classes_path):
        """Loads class names from a file"""
        with open(classes_path, "r") as f:
            return [line.strip() for line in f.readlines()]

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Displays image with bounding boxes, class names, and scores.
        """
        for box, box_class, box_score in zip(boxes, box_classes, box_scores):
            x1, y1, x2, y2 = box.astype(int)

            label = "{} {:.2f}".format(
                self.class_names[box_class],
                box_score
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),  # blue in BGR
                2
            )

            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),  # red in BGR
                1,
                cv2.LINE_AA
            )

        cv2.imshow(file_name, image)

        key = cv2.waitKey(0)

        if key == ord("s"):
            os.makedirs("detections", exist_ok=True)

            save_path = os.path.join(
                "detections",
                os.path.basename(file_name)
            )

            cv2.imwrite(save_path, image)

        cv2.destroyWindow(file_name)
