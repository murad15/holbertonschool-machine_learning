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
        classes_path: path to where the list of class names used
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
        """Loads the class names from a file"""
        with open(classes_path, "r") as f:
            return [line.strip() for line in f.readlines()]

    @staticmethod
    def load_images(folder_path):
        """
        Loads images from a folder.

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

    def preprocess_images(self, images):
        """
        Preprocesses images for the YOLO model.

        images: list of images as numpy.ndarrays

        Returns:
            pimages: numpy.ndarray of shape (ni, input_h, input_w, 3)
            image_shapes: numpy.ndarray of shape (ni, 2)
        """
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

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes
