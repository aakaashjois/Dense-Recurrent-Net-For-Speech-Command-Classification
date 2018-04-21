import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


class DatasetUtils:
    def __init__(self):
        self.PATH_TO_DATA = os.path.join(os.getcwd(), 'data')
        self.PATH_TO_AUDIO = os.path.join(self.PATH_TO_DATA, 'audio')

        self.keras = tf.keras
        self.label_encoder = self.get_label_encoder()

    def get_dataset_and_encoded_labels(self, dataset_path, labels_path):
        # Load dataset and labels from .npy files
        dataset = np.load(os.path.join(self.PATH_TO_DATA, dataset_path))
        # Reshape data for compatibility with Keras
        dataset = dataset.reshape((*dataset.shape, 1))
        labels = np.load(os.path.join(self.PATH_TO_DATA, labels_path))
        # One-hot encode the labels
        labels = self.keras.utils.to_categorical(self.label_encoder.transform(labels))

        return dataset, labels

    def get_label_encoder(self):
        ALL_LABELS = os.listdir(self.PATH_TO_AUDIO)
        INVALID_LABELS = ["bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow"]
        LABELS = list(set(ALL_LABELS) - set(INVALID_LABELS))
        LABELS.append("unknown")
        label_encoder = LabelEncoder().fit(LABELS)

        return label_encoder
