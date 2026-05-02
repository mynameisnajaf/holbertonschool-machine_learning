#!/usr/bin/env python3
"""A module that does the trick"""
import tensorflow as tf


class Yolo:
    """A class that does the trick"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initializer for the class"""
        self.model_path = tf.keras.models.load_model(model_path)
        with open(classes_path) as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
