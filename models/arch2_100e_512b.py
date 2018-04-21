import os
import time
import numpy as np
import tensorflow as tf

# Create keras links
keras = tf.keras
Input = keras.layers.Input
Conv2D = keras.layers.Conv2D
Flatten = keras.layers.Flatten
Dense = keras.layers.Dense
GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
Concatenate = keras.layers.Concatenate
Model = keras.Model


PATH_TO_DATA = os.path.join(os.getcwd(), 'data')
PATH_TO_AUDIO = os.path.join(PATH_TO_DATA, 'audio')

# Load dataset
train_data = np.load(os.path.join(PATH_TO_DATA, 'train_data.npy'))
train_labels = np.load(os.path.join(PATH_TO_DATA, 'train_labels.npy'))

validation_data = np.load(os.path.join(PATH_TO_DATA, 'validation_data.npy'))
validation_labels = np.load(os.path.join(PATH_TO_DATA, 'validation_labels.npy'))

test_data = np.load(os.path.join(PATH_TO_DATA, 'test_data.npy'))
test_labels = np.load(os.path.join(PATH_TO_DATA, 'test_labels.npy'))

# One-hot encode the labels
ALL_LABELS = os.listdir(PATH_TO_AUDIO)
INVALID_LABELS = ["bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow"]
LABELS = list(set(ALL_LABELS) - set(INVALID_LABELS))
LABELS.append("unknown")
NUM_CLASSES = len(LABELS)
ARCH_NAME = 'arch2_100e_512b'

label_encoder = LabelEncoder().fit(LABELS)

train_labels = keras.utils.to_categorical(label_encoder.transform(train_labels), num_classes=NUM_CLASSES)
validation_labels = keras.utils.to_categorical(label_encoder.transform(validation_labels), num_classes=NUM_CLASSES)
test_labels = keras.utils.to_categorical(label_encoder.transform(test_labels), num_classes=NUM_CLASSES)

# Reshape data for compatibility with Keras
train_data = train_data.reshape((*train_data.shape, 1))
validation_data = validation_data.reshape((*validation_data.shape, 1))
test_data = test_data.reshape((*test_data.shape, 1))

input_layer = Input(shape=train_data.shape[1:])
conv_1 = Conv2D(filters=48, kernel_size=(8, 3), padding='same', activation='relu')(input_layer)
conv_2 = Conv2D(filters=48, kernel_size=(8, 3), padding='same', activation='relu')(conv_1)
conv_3 = Conv2D(filters=36, kernel_size=(8, 3), padding='same', activation='relu')(conv_2)
conv_4 = Conv2D(filters=36, kernel_size=(8, 3), padding='same', activation='relu')(conv_3)
gap_1 = GlobalAveragePooling2D()(conv_4)
dense = Dense(30, activation='softmax')(gap_1)
model = Model(inputs=input_layer, outputs=dense)

# Display model summary for reference
model.summary()

early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                    min_delta=0, 
                                                    patience=10, 
                                                    verbose=0, 
                                                    mode='auto')

reduce_lr_plateau_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                               factor=0.1, 
                                                               patience=5, 
                                                               verbose=0, 
                                                               mode='auto', 
                                                               cooldown=0, min_lr=0)
csv_logger = keras.callbacks.CSVLogger(ARCH_NAME + '.csv')

callbacks = [early_stop_callback, reduce_lr_plateau_callback, csv_logger]

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
start_time == time.time()
history = model.fit(train_data, 
          train_labels, 
          epochs=100, 
          batch_size=512, 
          validation_data=(validation_data, validation_labels), 
          shuffle=True,
          callbacks=callbacks,
          verbose=1)
stop_time = time.time()
run_time = stop_time - start_time

model.save(ARCH_NAME + +'_' + str(run_time) + 't' + '.h5')