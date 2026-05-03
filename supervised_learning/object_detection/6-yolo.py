#!/usr/bin/env python3
"""YOLO object detection"""

import os
import cv2
import numpy as np
from tensorflow import keras


class Yolo:
    """Uses the YOLO v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initializes the YOLO object detector"""

        self.model = keras.models.load_model(model_path)
        self.class_names = self.load_classes(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def load_classes(classes_path):
        """Loads the class names from a file"""
        with open(classes_path, "r") as f:
            return [line.strip() for line in f.readlines()]

    @staticmethod
    def load_images(folder_path):
        """Loads all images from a folder"""
        images = []
        image_paths = []

        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            if image is not None:
                images.append(image)
                image_paths.append(image_path)

        return images, image_paths

    def preprocess_images(self, images):
        """Preprocesses images for the YOLO model"""
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for image in images:
            image_shapes.append(image.shape[:2])

            resized = cv2.resize(
                image,
                (input_w, input_h),
                interpolation=cv2.INTER_CUBIC
            )

            resized = resized / 255
            pimages.append(resized)

        return np.array(pimages), np.array(image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Displays an image with bounding boxes, class names, and scores.

        image: numpy.ndarray containing an unprocessed image
        boxes: numpy.ndarray containing boundary boxes
        box_classes: numpy.ndarray containing class indices for each box
        box_scores: numpy.ndarray containing box scores for each box
        file_name: file path where the original image is stored
        """
        img = image.copy()

        for box, box_class, box_score in zip(boxes, box_classes, box_scores):
            x1, y1, x2, y2 = box.astype(int)

            class_name = self.class_names[box_class]
            label = "{} {:.2f}".format(class_name, box_score)

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            cv2.putText(
                img,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        cv2.imshow(file_name, img)

        key = cv2.waitKey(0)

        if key == ord("s"):
            if not os.path.exists("detections"):
                os.makedirs("detections")

            save_path = os.path.join("detections", os.path.basename(file_name))
            cv2.imwrite(save_path, img)

        cv2.destroyWindow(file_name)
