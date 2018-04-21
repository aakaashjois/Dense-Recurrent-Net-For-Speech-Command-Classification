import os
import time
import tensorflow as tf
import source.utils as utils

ARCH_NAME = os.path.basename(__file__).split(".")[0]


def get_keras_model():
    # Create keras links
    keras = tf.keras
    Input = keras.layers.Input
    Conv2D = keras.layers.Conv2D
    Dense = keras.layers.Dense
    GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
    Concatenate = keras.layers.Concatenate
    Model = keras.Model

    input_layer = Input(shape=train_data.shape[1:])
    conv_1 = Conv2D(filters=48, kernel_size=(8, 3), padding='same', activation='relu')(input_layer)
    conv_2 = Conv2D(filters=48, kernel_size=(8, 3), padding='same', activation='relu')(conv_1)
    conv_3 = Conv2D(filters=36, kernel_size=(8, 3), padding='same', activation='relu')(conv_2)
    concat = Concatenate(axis=3)([conv_3, conv_1])
    conv_4 = Conv2D(filters=48, kernel_size=(8, 3), padding='same', activation='relu')(concat)
    conv_5 = Conv2D(filters=36, kernel_size=(8, 3), padding='same', activation='relu')(conv_4)
    gap_1 = GlobalAveragePooling2D()(conv_5)
    dense = Dense(30, activation='softmax')(gap_1)
    model = Model(inputs=input_layer, outputs=dense)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Load dataset and one-hot encoded labels
dataset_utils = utils.DatasetUtils()
train_data, train_labels = dataset_utils.get_dataset_and_encoded_labels('train_data.npy', 'train_labels.npy')
validation_data, validation_labels = dataset_utils.get_dataset_and_encoded_labels('validation_data.npy',
                                                                                  'validation_labels.npy')
test_data, test_labels = dataset_utils.get_dataset_and_encoded_labels('test_data.npy', 'test_labels.npy')

model = get_keras_model()
keras_utils = utils.KerasUtils()
model.summary()

start_time = time.time()
model.fit(train_data,
          train_labels,
          epochs=100,
          batch_size=512,
          validation_data=(validation_data, validation_labels),
          shuffle=True,
          class_weights=dataset_utils.get_class_weights(train_labels),
          callbacks=keras_utils.get_keras_callbacks(ARCH_NAME),
          verbose=1)
stop_time = time.time()
run_time = stop_time - start_time
model.save(ARCH_NAME + '_' + str(run_time) + 't' + '.h5')
print('Finished training model. Took {} ms'.format(run_time), flush=True)
