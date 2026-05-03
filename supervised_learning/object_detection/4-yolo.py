#!/usr/bin/env python3
"""YOLO object detection"""

import os
import cv2
import numpy as np


class Yolo:
    """Uses the YOLO v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        model_path: path to where a Darknet Keras model is stored
        classes_path: path to where the list of class names
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
        """Loads class names from a file"""
        with open(classes_path, "r") as f:
            return [line.strip() for line in f.readlines()]

    @staticmethod
    def load_images(folder_path):
        """
        Loads all images from a folder.

        folder_path: path to the folder containing images

        Returns:
            images: list of images as numpy.ndarrays
            image_paths: list of image paths
        """
        images = []
        image_paths = []

        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            if image is not None:
                images.append(image)
                image_paths.append(image_path)

        return images, image_paths
