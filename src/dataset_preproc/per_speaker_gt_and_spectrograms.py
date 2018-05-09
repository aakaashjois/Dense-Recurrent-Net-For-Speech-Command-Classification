import os

import numpy as np
from tqdm import tqdm

tqdm.monitor_interval = 0
import librosa

SAMPLING_RATE = 16000
PATH_TO_AUDIO = '../data/audio/'
PATH_TO_SAVE = '../data/speaker_spectrograms/'
INVALID_LABELS = ['bed', 'bird', 'cat', 'dog', 'happy', 'house', 'marvin', 'sheila', 'tree', 'wow]
categories = os.listdir(PATH_TO_AUDIO)
print('Number of categories: {}'.format(len(categories)))

if not os.path.exists(PATH_TO_SAVE):
    os.mkdir(PATH_TO_SAVE)

# List of speakers
speaker_names = []
# List of all filenames in the dataset
all_file_names = []
# Key,value pairs of speaker, filenames for each speaker
filenames_per_speaker = {}
# Key,value pairs of speaker, ground truth for each speaker
gt_per_speaker = {}

def _get_speaker_name_find_gt_find_all_audio(category, filename):
    '''
    - Returns speaker name from file name
    - Appends all audio files belonging to this speaker to dict
    - Makes ground truth values belonging to this speaker in another dict
    '''
    speaker = filename.split('_')[0]

    # Append all audio files belonging to this speaker to dict
    if speaker in filenames_per_speaker:
        filenames_this_speaker = filenames_per_speaker[speaker]
    else:
        filenames_this_speaker = []
    filenames_this_speaker.append(os.path.join(category, filename))
    filenames_per_speaker[speaker] = filenames_this_speaker

    # Make ground truth values belonging to this speaker in another dict
    if category in INVALID_LABELS:
        category = 'unknown'
    if speaker in gt_per_speaker:
        gt_this_speaker = gt_per_speaker[speaker]
    else:
        gt_this_speaker = []
    gt_this_speaker.append(category)
    gt_per_speaker[speaker] = gt_this_speaker
    return speaker


def _read_audio_file(audio_file_path):
    """Read the .wav file at {audio_file_path} and return it as a numpy array
    Arguments:
        audio_file_path {str} -- absolute path to audio file
    Returns:
        numpy.ndarray -- The audio file loaded as a numpy array
    """
    # Read audio file
    # If audio file is shorter than 16000 samples, zero-pad keeping the audio in the center
    # If audio file is longer than 16000 samples, center crop to 16000 samples
    audio_file = librosa.load(os.path.join(PATH_TO_AUDIO, audio_file_path), sr=SAMPLING_RATE)[0]
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


print('Retrieving speaker names...')
for c in tqdm(categories):
    category_path = os.path.join(PATH_TO_AUDIO, c)

    # List of audio filenames in category c; append to all_file_names
    files_in_category = os.listdir(category_path)
    all_file_names.extend(files_in_category)

    # Retrieve speaker name
    speaker_name = np.vectorize(_get_speaker_name_find_gt_find_all_audio)(c, files_in_category)
    speaker_names.extend(speaker_name)

# Retrieve only unique speaker_names (list contains repeats of speaker names across entire dataset)
speaker_names = np.unique(speaker_names)
# Convert all_file_names to a numpy.ndarray
all_file_names = np.array(all_file_names)
print('Number of speakers: {}'.format(len(speaker_names)))
print('Total number of files: {}'.format(len(all_file_names)))

print('Generating spectrograms per speaker; saving to disk...')
for k, v in tqdm(filenames_per_speaker.items()):
    # key-value pairs of speaker name-files belonging to speaker
    spectrograms_for_this_speaker = []
    for x in v:
        spectrograms_for_this_speaker.append(_get_mel_power_spectrogram(_read_audio_file(x)))
    np.save(k, os.path.join(PATH_TO_SAVE, spectrograms_for_this_speaker))

print('Done.)')
