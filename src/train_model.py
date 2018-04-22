import argparse
import os
import time
import tensorflow as tf
from utils import utils
from utils.generate_model import ModelGenerator

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--architecture', help='compiles model of given architecture number [options: 1 or 2]', type=int)
parser.add_argument('-e', '--epochs', help='runs the model for given number of epochs', type=int)
parser.add_argument('-b', '--batch_size', help='runs the model with given batch size', type=int)
args = parser.parse_args()

ARCH_NAME = 'arch' + str(args.architecture) + '_' + str(args.epochs) + 'e' + '_' + str(args.batch_size) + 'b' 

# Load dataset and one-hot encoded labels
dataset_utils = utils.DatasetUtils()
train_data, train_labels, train_weights = dataset_utils.get_dataset_and_encoded_labels('train_data.npy', 'train_labels.npy', get_weights=True)
validation_data, validation_labels = dataset_utils.get_dataset_and_encoded_labels('validation_data.npy',
                                                                                  'validation_labels.npy')
test_data, test_labels = dataset_utils.get_dataset_and_encoded_labels('test_data.npy', 'test_labels.npy')

model_generator = ModelGenerator(args.architecture)
model = model_generator.generate()
keras_utils = utils.KerasUtils()
model.summary()

start_time = time.time()
model.fit(train_data,
          train_labels,
          epochs=100,
          batch_size=512,
          validation_data=(validation_data, validation_labels),
          shuffle=True,
          class_weight=train_weights,
          callbacks=keras_utils.get_keras_callbacks(ARCH_NAME),
          verbose=1)
stop_time = time.time()
run_time = stop_time - start_time
print('Finished training model. Took {} s'.format(run_time), flush=True)
