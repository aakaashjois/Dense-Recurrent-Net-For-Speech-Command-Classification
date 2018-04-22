import os

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


class DatasetUtils:
    def __init__(self):
        self.PATH_TO_DATA = os.path.join(os.getcwd(), 'data')
        self.PATH_TO_AUDIO = os.path.join(self.PATH_TO_DATA, 'audio')

        self.keras = tf.keras
        self.label_encoder = self.get_label_encoder()

    def get_dataset_and_encoded_labels(self, dataset_path, labels_path, get_weights=False):
        # Load dataset and labels from .npy files
        dataset = np.load(os.path.join(self.PATH_TO_DATA, dataset_path))
        # Reshape data for compatibility with Keras
        dataset = dataset.reshape((*dataset.shape, 1))
        labels = np.load(os.path.join(self.PATH_TO_DATA, labels_path))
        labels = self.label_encoder.transform(labels)
        if get_weights:
            weights_dict = self.get_class_weights(labels)
            labels = self.keras.utils.to_categorical(labels)
            return dataset, labels, weights_dict
        # One-hot encode the labels
        labels = self.keras.utils.to_categorical(labels)
        return dataset, labels

    def get_label_encoder(self):
        ALL_LABELS = os.listdir(self.PATH_TO_AUDIO)
        INVALID_LABELS = ["bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow"]
        LABELS = list(set(ALL_LABELS) - set(INVALID_LABELS))
        LABELS.append("unknown")
        label_encoder = LabelEncoder().fit(LABELS)

        return label_encoder

    def get_class_weights(self, labels):
        # Generate class weights as described by Chris Dinant at https://github.com/chrisdinant/speech/blob/master/train.ipynb
        uniques, count = np.unique(labels, return_counts=True)
        count = count / max(count)
        return dict(zip(uniques.astype(int), count))