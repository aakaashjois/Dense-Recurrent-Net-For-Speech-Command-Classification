import argparse
import time

from model_generator import ModelGenerator
from utils import DatasetUtils

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-a', '--architecture', help='Compiles Keras model of architecture number [options: 1, 2 or 3]',
                    type=int)
parser.add_argument('-e', '--epochs', help='Number of training epochs', type=int)
parser.add_argument('-b', '--batch_size', help='Training batch size', type=int)
args = parser.parse_args()

ARCH_NAME = 'arch' + str(args.architecture) + '_' + str(args.epochs) + 'e' + '_' + str(args.batch_size) + 'b'

# Load dataset and one-hot encoded labels
dataset_utils = DatasetUtils()
train_data, train_labels, train_weights = dataset_utils.get_dataset_and_encoded_labels('train_data.npy',
                                                                                       'train_labels.npy',
                                                                                       get_weights=True)
validation_data, validation_labels = dataset_utils.get_dataset_and_encoded_labels('validation_data.npy',
                                                                                  'validation_labels.npy')
test_data, test_labels = dataset_utils.get_dataset_and_encoded_labels('test_data.npy', 'test_labels.npy')

# Get Keras model and show summary
model_generator = ModelGenerator()
model = model_generator.get_keras_model(args.architecture, train_data.shape[1:])
model.summary()

# Train model and log training time
start_time = time.time()

model.fit(train_data,
          train_labels,
          epochs=args.epochs,
          batch_size=args.batch_size,
          validation_data=(validation_data, validation_labels),
          shuffle=True,
          class_weight=train_weights,
          callbacks=model_generator.get_keras_callbacks(ARCH_NAME),
          verbose=1)

stop_time = time.time()
run_time = stop_time - start_time
print('Finished training model in {} s.'.format(run_time), flush=True)
