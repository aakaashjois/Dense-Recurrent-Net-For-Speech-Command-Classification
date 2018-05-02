import glob
import os

import jams
import librosa
import muda
import numpy as np
from scipy.io.wavfile import write as scipy_write
from tqdm import tqdm
import argparse


class CreateDataset():
    def __init__(self):
        self.PATH_TO_DATA = os.path.join(os.getcwd(), 'data')
        self.PATH_TO_AUDIO = os.path.join(self.PATH_TO_DATA, 'audio')
        self.SAMPLING_RATE = 16000
        self.INVALID_LABELS = ["bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow"]
        self.JAM_OBJECT = jams.JAMS()

        # Make synthetic noise of SAMPLING_RATE samples form a Gaussian distribution
        synthetic_noise = np.random.normal(size=self.SAMPLING_RATE)
        # Write synthetic noise to disk
        scipy_write('synthetic_noise.wav', self.SAMPLING_RATE, synthetic_noise)
        self.noise_deformer = muda.deformers.BackgroundNoise(n_samples=1, files='synthetic_noise.wav', weight_max=0.101)

    def _read_audio_file(self, audio_file_path):
        """Read the .wav file at {audio_file_path} and return it as a numpy array

        Arguments:
            audio_file_path {str} -- absolute path to audio file

        Returns:
            numpy.ndarray -- The audio file loaded as a numpy array
        """

        # Read audio file
        # If audio file is shroter than 16000 samples, zero-pad keeping the audio in the center
        # If audio file is longer than 16000 samples, center crop to 16000 samples
        audio_file = librosa.load(os.path.join(self.PATH_TO_AUDIO, audio_file_path), sr=SAMPLING_RATE)[0]
        if len(audio_file) < self.SAMPLING_RATE:
            if len(audio_file) % 2 == 1:
                audio_file = np.append(audio_file, [0])
            pad_width = (self.SAMPLING_RATE - len(audio_file)) // 2
            audio_file = np.pad(audio_file, pad_width=pad_width, mode='constant')
            audio_file = audio_file[0: self.SAMPLING_RATE]
        elif len(audio_file) > self.SAMPLING_RATE:
            length_to_truncate = (len(audio_file) - self.SAMPLING_RATE) // 2
            audio_file = audio_file[length_to_truncate: self.SAMPLING_RATE + length_to_truncate]
        return audio_file

    def _get_mel_power_spectrogram(self, audio_file, n_fft=1024, hop_length=256, fmax=3000):
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
                                                                  sr=self.SAMPLING_RATE,
                                                                  n_fft=n_fft,
                                                                  hop_length=hop_length,
                                                                  fmax=fmax),
                                   ref=np.max)

    def _return_valid_label(self, audio_label):
        """Check if label of audio file is valid

        Arguments:
            audio_label {str} -- The label of audio file

        Returns:
            {str} -- The valid label
        """

        return "unknown" if audio_label in self.INVALID_LABELS else audio_label

    def _make_noisy(self, audio):
        if np.random.uniform() > 0.5:
            # Create blank JAM variable
            jam_packed_audio = muda.jam_pack(self.JAM_OBJECT, _audio=dict(y=audio, sr=self.SAMPLING_RATE))
            output_jam = [x for x in self.noise_deformer.transform(jam_packed_audio)]
            output = output_jam[0].sandbox.muda._audio['y']
            sr = output_jam[0].sandbox.muda._audio['sr']
            output = librosa.core.resample(output, sr, self.SAMPLING_RATE)
            return output
        else:
            return audio

    def create_mel_spectrograms(self, train_data_paths, validation_data_paths, test_data_paths, noisy):
        """Split the data into train, validation and test
        Corrupt train data with noise; train samples to corrupt are chosen from a Gaussian distribution

        Returns:
            tuple -- Tuple of train data (clean and noisy), validation data and test data and their
                    corresponding labels
        """
        print('Creating Mel-power spectrograms', flush=True)

        # Empty variables for data
        train_data = []
        validation_data = []
        test_data = []

        # Empty variables for labels
        train_labels = []
        train_labels_unrolled = []
        validation_labels = []
        validation_labels_unrolled = []
        test_labels = []
        test_labels_unrolled = []

        if noisy:
            print('Creating partially corrupted noisy train data...', flush=True)
            for path in tqdm(train_data_paths):
                train_data.append(self._get_mel_power_spectrogram(self._make_noisy(self._read_audio_file(path))))
        else:
            print('Creating clean train data...', flush=True)
            for path in tqdm(train_data_paths):
                train_data.append(self._get_mel_power_spectrogram((self._read_audio_file(path))))

        print('Creating validation data...', flush=True)
        for path in tqdm(validation_data_paths):
            validation_data.append(self._get_mel_power_spectrogram(self._read_audio_file(path)))

        print('Creating test data...', flush=True)
        for path in tqdm(test_data_paths):
            test_data.append(self._get_mel_power_spectrogram(self._read_audio_file(path)))

        print('Creating labels for train data...', flush=True)
        for path in tqdm(train_data_paths):
            train_labels_unrolled.append(path.split(os.path.sep)[0])
            train_labels.append(self._return_valid_label(path.split(os.path.sep)[0]))

        print('Creating labels for validation data...', flush=True)
        for path in tqdm(validation_data_paths):
            validation_labels_unrolled.append(path.split(os.path.sep)[0])
            validation_labels.append(self._return_valid_label(path.split(os.path.sep)[0]))

        print('Creating labels for test data...', flush=True)
        for path in tqdm(test_data_paths):
            test_labels_unrolled.append(path.split(os.path.sep)[0])
            test_labels.append(self._return_valid_label(path.split(os.path.sep)[0]))

        all_train = train_data, train_labels, train_labels_unrolled
        all_validation = validation_data, validation_labels, validation_labels_unrolled
        all_test = test_data, test_labels, test_labels_unrolled

        return all_train, all_validation, all_test

    def save_spectrograms_to_disk(self, all_train, all_validation, all_test):
        train_data, train_labels, train_labels_unrolled = all_train
        validation_data, validation_labels, validation_labels_unrolled = all_validation
        test_data, test_labels, test_labels_unrolled = all_test

        print('Normalizing all data...', flush=True)
        train_data = (np.array(train_data) - np.mean(train_data)) / np.std(train_data)
        validation_data = (np.array(validation_data) - np.mean(validation_data)) / np.std(validation_data)
        test_data = (np.array(test_data) - np.mean(test_data)) / np.std(test_data)

        # Save data to disk
        print('Saving data to ' + self.PATH_TO_DATA)
        np.save(os.path.join(self.PATH_TO_DATA, 'train_data'), train_data)
        np.save(os.path.join(self.PATH_TO_DATA, 'validation_data'), validation_data)
        np.save(os.path.join(self.PATH_TO_DATA, 'test_data'), test_data)

        # Save labels to disk
        np.save(os.path.join(self.PATH_TO_DATA, 'train_labels'), train_labels)
        np.save(os.path.join(self.PATH_TO_DATA, 'train_labels_unrolled'), train_labels_unrolled)
        np.save(os.path.join(self.PATH_TO_DATA, 'validation_labels'), validation_labels)
        np.save(os.path.join(self.PATH_TO_DATA, 'validation_labels_unrolled'), validation_labels_unrolled)
        np.save(os.path.join(self.PATH_TO_DATA, 'test_labels'), test_labels)
        np.save(os.path.join(self.PATH_TO_DATA, 'test_labels_unrolled'), test_labels_unrolled)


noisy = False
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('noisy', help='Corrupt some of the data with random noise', type=bool, required=True)
    args = parser.parse_args()
    noisy = args['noisy']

dataset_creator = CreateDataset()

# Removing leading '/data/audio/' from all paths
print('Finding path to all audio', flush=True)
all_data_paths = glob.glob(os.path.join(dataset_creator.PATH_TO_AUDIO, '*', '*'))
all_data_paths = np.vectorize(str.replace)(all_data_paths, os.path.join(dataset_creator.PATH_TO_AUDIO, ''), '')

# Create a lambda function that helps in vectorizing the string replace function
split_join = lambda x: os.path.join(*str.split(x, '/'))

print('Finding path to all testing data', flush=True)
with open(os.path.join(dataset_creator.PATH_TO_DATA, 'testing_list.txt')) as f:
    test_data_paths = f.readlines()
test_data_paths = np.vectorize(str.replace)(test_data_paths, '\n', '')
test_data_paths = np.vectorize(split_join)(test_data_paths)

print('Finding path to all validation data', flush=True)
with open(os.path.join(dataset_creator.PATH_TO_DATA, 'validation_list.txt')) as f:
    validation_data_paths = f.readlines()
validation_data_paths = np.vectorize(str.replace)(validation_data_paths, '\n', '')
validation_data_paths = np.vectorize(split_join)(validation_data_paths)

print('Finding path to all training data', flush=True)
train_data_paths = list(set(all_data_paths) ^ set(validation_data_paths) ^ set(test_data_paths))

all_train, all_validation, all_test = dataset_creator.create_mel_spectrograms(train_data_paths, validation_data_paths,
                                                                              test_data_paths, noisy=noisy)
dataset_creator.save_spectrograms_to_disk(all_train, all_validation, all_test)
print('Done', flush=True)
