import glob
import os

import librosa
import numpy as np
from tqdm import tqdm

PATH_TO_DATA = os.path.join(os.getcwd(), 'data')
PATH_TO_AUDIO = os.path.join(PATH_TO_DATA, 'audio')
SAMPLING_RATE = 16000
INVALID_LABELS = ["bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow"]


def _read_audio_file(audio_file_path):
    """Read the .wav file at {audio_file_path} and return it as a numpy array

    Arguments:
        audio_file_path {str} -- absolute path to audio file

    Returns:
        numpy.ndarray -- The audio file loaded as a numpy array
    """

    # Read audio file
    # If audio file is shroter than 16000 samples, zero-pad keeping the audio in the center
    # If audio file is longer than 16000 samples, center crop to 16000 samples
    audio_file = librosa.load(os.path.join(PATH_TO_AUDIO, audio_file_path))[0]
    if len(audio_file) < SAMPLING_RATE:
        if len(audio_file) % 2 == 1:
            audio_file = np.append(audio_file, [0])
        pad_width = (SAMPLING_RATE - len(audio_file)) // 2
        audio_file = np.pad(audio_file, pad_width=pad_width, mode='constant')
        audio_file = audio_file[0: SAMPLING_RATE]
    elif len(audio_file) > SAMPLING_RATE:
        length_to_truncate = (len(audio_file) - SAMPLING_RATE) // 2
        audio_file = audio_file[length_to_truncate: SAMPLING_RATE + length_to_truncate]
    return audio_file


def _get_mel_power_spectrogram(audio_file, n_fft=1024, hop_length=256, fmax=3000):
    """Generates mel power spectrogram of the {audio_file}

    Arguments:
        audio_file {numpy.ndarray} -- The audio file

    Keyword Arguments:
        n_fft {int} -- Length of FFT (default: {1024})
        hop_length {int} -- The hop length (default: {256})
        fmax {int} -- The maximum frequency of the mel filterbank (default: {3000})

    Returns:
        numpy.ndarray -- Mel power spectrogram
    """
    return librosa.power_to_db(librosa.feature.melspectrogram(audio_file,
                                                              sr=SAMPLING_RATE,
                                                              n_fft=n_fft,
                                                              hop_length=hop_length,
                                                              fmax=fmax),
                               ref=np.max)


def _return_valid_label(audio_label):
    """Check if label of audio file is valid

    Arguments:
        audio_label {str} -- The label of audio file

    Returns:
        {str} -- The valid label
    """

    return "unknown" if audio_label in INVALID_LABELS else audio_label


def create_mel_spectrograms():
    """Split the data into train, validation and test and save to disk as numpy arrays
    """
    print('Creating Mel-power spectrograms', flush=True)

    train_data = []
    train_labels = []
    train_labels_unrolled = []
    validation_data = []
    validation_labels = []
    validation_labels_unrolled = []
    test_data = []
    test_labels = []
    test_labels_unrolled = []

    print('Creating train data', flush=True)
    for path in tqdm(train_data_paths):
        train_data.append(_get_mel_power_spectrogram(_read_audio_file(path)))
        train_labels_unrolled.append(path.split(os.path.sep)[0])
        train_labels.append(_return_valid_label(path.split(os.path.sep)[0]))

    print('Creating validation data', flush=True)
    for path in tqdm(validation_data_paths):
        validation_data.append(_get_mel_power_spectrogram(_read_audio_file(path)))
        validation_labels_unrolled.append(path.split(os.path.sep)[0])
        validation_labels.append(_return_valid_label(path.split(os.path.sep)[0]))

    print('Creating test data', flush=True)
    for path in tqdm(test_data_paths):
        test_data.append(_get_mel_power_spectrogram(_read_audio_file(path)))
        test_labels_unrolled(path.split(os.path.sep)[0])
        test_labels.append(_return_valid_label(path.split(os.path.sep)[0]))

    print('Normalizing the data', flush=True)
    train_data = (np.array(train_data) - np.mean(train_data)) / np.std(train_data)
    validation_data = (np.array(validation_data) - np.mean(validation_data)) / np.std(validation_data)
    test_data = (np.array(test_data) - np.mean(test_data)) / np.std(test_data)

    print('Saving the data in ' + PATH_TO_DATA)
    np.save(os.path.join(PATH_TO_DATA, 'train_data'), train_data)
    np.save(os.path.join(PATH_TO_DATA, 'train_labels'), train_labels)
    np.save(os.path.join(PATH_TO_DATA, 'train_labels_unrolled'), train_labels_unrolled)
    np.save(os.path.join(PATH_TO_DATA, 'validation_data'), validation_data)
    np.save(os.path.join(PATH_TO_DATA, 'validation_labels'), validation_labels)
    np.save(os.path.join(PATH_TO_DATA, 'validation_labels_unrolled'), validation_labels_unrolled)
    np.save(os.path.join(PATH_TO_DATA, 'test_data'), test_data)
    np.save(os.path.join(PATH_TO_DATA, 'test_labels'), test_labels)
    np.save(os.path.join(PATH_TO_DATA, 'test_labels_unrolled'), test_labels_unrolled)


# Removing leading '/data/audio/' from all paths
print('Finding the path of all audio files', flush=True)
all_data_paths = glob.glob(os.path.join(PATH_TO_AUDIO, '*', '*'))
all_data_paths = np.vectorize(str.replace)(all_data_paths, os.path.join(PATH_TO_AUDIO, ''), '')

# Create a lambda function that helps in vectorizing the string replace function
split_join = lambda x: os.path.join(*str.split(x, '/'))

print('Finding the path for all testing data', flush=True)
with open(os.path.join(PATH_TO_DATA, 'testing_list.txt')) as f:
    test_data_paths = f.readlines()
test_data_paths = np.vectorize(str.replace)(test_data_paths, '\n', '')
test_data_paths = np.vectorize(split_join)(test_data_paths)

print('Finding the path for all validation data', flush=True)
with open(os.path.join(PATH_TO_DATA, 'validation_list.txt')) as f:
    validation_data_paths = f.readlines()
validation_data_paths = np.vectorize(str.replace)(validation_data_paths, '\n', '')
validation_data_paths = np.vectorize(split_join)(validation_data_paths)

print('Finding the path for all training data', flush=True)
train_data_paths = list(set(all_data_paths) ^ set(validation_data_paths) ^ set(test_data_paths))

create_mel_spectrograms()
print('Done', flush=True)
