import os
import tensorflow as tf


class ModelGenerator:
    def __init__(self):
        self.PATH_TO_MODELS = os.path.join(os.getcwd(), 'models')
        self.Input = tf.keras.layers.Input
        self.Conv2D = tf.keras.layers.Conv2D
        self.Dense = tf.keras.layers.Dense
        self.GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
        self.Concatenate = tf.keras.layers.Concatenate
        self.Model = tf.keras.Model

    def get_keras_model(self, architecture, input_shape):
        """Make Keras model based on user-specified architecture.

        Arguments:
            architecture {int} -- model architecture type
            input_shape {tuple} -- shape of input to Keras model

        Returns:
            tf.keras.Model -- Keras Model (functional API)
        """
        if type(architecture) is not int:
            raise ValueError("Invalid argument architecture. Expected argument of int type, got {} instead.".format(
                type(architecture)))
        if len(input_shape) != 3:
            raise ValueError(
                "Invalid argument input_shape. Expected length 3, got {} instead.".format(len(input_shape)))

        if architecture == 1:
            return self._architecture_1_model(input_shape)
        elif architecture == 2:
            return self.architecture_2_model(input_shape)
        else:
            raise ValueError('Unknown architecture.')

    def _architecture_1_model(self, input_shape):
        input_layer = self.Input(shape=input_shape)
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

    def architecture_2_model(self, input_shape):
        input_layer = self.Input(shape=input_shape)
        conv_1 = self.Conv2D(filters=48, kernel_size=(8, 3), padding='same', activation='relu')(input_layer)
        conv_2 = self.Conv2D(filters=48, kernel_size=(8, 3), padding='same', activation='relu')(conv_1)
        conv_3 = self.Conv2D(filters=36, kernel_size=(8, 3), padding='same', activation='relu')(conv_2)
        conv_4 = self.Conv2D(filters=36, kernel_size=(8, 3), padding='same', activation='relu')(conv_3)
        gap_1 = self.GlobalAveragePooling2D()(conv_4)
        dense = self.Dense(21, activation='softmax')(gap_1)
        model = self.Model(inputs=input_layer, outputs=dense)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

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

        history_logger = self.callbacks.CSVLogger(os.path.join(self.PATH_TO_MODELS, ARCH_NAME) + '.csv')

        best_model = self.callbacks.ModelCheckpoint(filepath=os.path.join(self.PATH_TO_MODELS, ARCH_NAME) + '.h5',
                                                    verbose=0,
                                                    save_best_only=True)

        return [early_stop_callback, reduce_lr_plateau_callback, history_logger, best_model]
