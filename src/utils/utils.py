import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from collections import Counter


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

    def get_class_weights(self, labels):
        # Generate class weights as described by Chris Dinant at https://github.com/chrisdinant/speech/blob/master/train.ipynb
        counter = Counter(labels)
        majority = max(counter.values)
        return {cls: float(majority / count) for cls, count in counter.items()}


class KerasUtils:
    def __init__(self):
        self.callbacks = tf.keras.callbacks

    def get_keras_callbacks(self, ARCH_NAME):
        early_stop_callback = self.callbacks.EarlyStopping(monitor='val_loss',
                                                           min_delta=0,
                                                           patience=10,
                                                           verbose=0,
                                                           mode='auto')

        reduce_lr_plateau_callback = self.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                      factor=0.1,
                                                                      patience=5,
                                                                      verbose=0,
                                                                      mode='auto',
                                                                      cooldown=0, min_lr=0)
        history_logger = self.callbacks.CSVLogger(ARCH_NAME + '.csv')
        best_model = self.callbacks.ModelCheckpoint(filepath=ARCH_NAME + '.h5',
                                                    verbose=0,
                                                    save_best_only=True)
        return [early_stop_callback, reduce_lr_plateau_callback, history_logger, best_model]
