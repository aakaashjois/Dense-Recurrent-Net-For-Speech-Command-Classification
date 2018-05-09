import os

import numpy as np
import tensorflow as tf

keras = tf.keras

SAMPLING_RATE = 16000
PATH_TO_ALL_SPEAKERS = '../data/speaker_spectrograms/'
PATH_TO_SAVE = '../data/speaker_predictions/'
PATH_TO_MODEL = '../models/arch3_100e_512b.h5'

all_speaker_files = os.listdir(PATH_TO_ALL_SPEAKERS)
model = keras.models.load_model(PATH_TO_MODEL)

print('Get classification predictions for each speaker; save to disk...')
print('Number of speakers: {}'.format(len(all_speaker_files)))
for index, speaker_name in enumerate(all_speaker_files):
    print('{}, speaker: {}'.format(index, speaker_name))
    speaker_spectrograms = np.load(os.path.join(PATH_TO_ALL_SPEAKERS, speaker_name))
    speaker_spectrograms = np.reshape(speaker_spectrograms, speaker_spectrograms.shape + (1,))
    y_preds = model.predict(speaker_spectrograms)
    np.save(y_preds, os.path.join(PATH_TO_SAVE, speaker_name))

print('Done.')
