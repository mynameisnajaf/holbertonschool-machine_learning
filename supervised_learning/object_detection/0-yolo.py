#!/usr/bin/env python3
"""A module that does the trick"""
import tensorflow as tf


class Yolo:
    """A class that does the trick"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initializer for the class"""
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
