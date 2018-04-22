import tensorflow as tf
import sys


class ModelGenerator:
    def __init__(self, architecture):
        self.Input = tf.keras.layers.Input
        self.Conv2D = tf.keras.layers.Conv2D
        self.Dense = tf.keras.layers.Dense
        self.GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
        self.Concatenate = tf.keras.layers.Concatenate
        self.Model = tf.keras.Model
        self.architecture = architecture

    
    def generate():
        if self.architecture == 1:
            return self.architecture_1_model()
        elif self.architecture == 2:
            return self.architecture_2_model()
        else:
            raise ValueError('Unknown architecture.')


    def architecture_1_model(self):
        input_layer = self.Input(shape=train_data.shape[1:])
        conv_1 = self.Conv2D(filters=48, kernel_size=(8, 3), padding='same', activation='relu')(input_layer)
        conv_2 = self.Conv2D(filters=48, kernel_size=(8, 3), padding='same', activation='relu')(conv_1)
        conv_3 = self.Conv2D(filters=36, kernel_size=(8, 3), padding='same', activation='relu')(conv_2)
        concat = self.Concatenate(axis=3)([conv_3, conv_1])
        conv_4 = self.Conv2D(filters=48, kernel_size=(8, 3), padding='same', activation='relu')(concat)
        conv_5 = self.Conv2D(filters=36, kernel_size=(8, 3), padding='same', activation='relu')(conv_4)
        gap_1 = self.GlobalAveragePooling2D()(conv_5)
        dense = self.Dense(21, activation='softmax')(gap_1)
        model = self.Model(inputs=input_layer, outputs=dense)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model


    def architecture_2_model(self):
        input_layer = Input(shape=train_data.shape[1:])
        conv_1 = Conv2D(filters=48, kernel_size=(8, 3), padding='same', activation='relu')(input_layer)
        conv_2 = Conv2D(filters=48, kernel_size=(8, 3), padding='same', activation='relu')(conv_1)
        conv_3 = Conv2D(filters=36, kernel_size=(8, 3), padding='same', activation='relu')(conv_2)
        conv_4 = Conv2D(filters=36, kernel_size=(8, 3), padding='same', activation='relu')(conv_3)
        gap_1 = GlobalAveragePooling2D()(conv_4)
        dense = Dense(21, activation='softmax')(gap_1)
        model = Model(inputs=input_layer, outputs=dense)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model