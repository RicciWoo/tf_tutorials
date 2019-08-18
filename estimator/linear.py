# Setup
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow.feature_column as fc

import os
import sys

import matplotlib.pyplot as plt
from IPython.display import clear_output

# enable eager execution to inspect this program as we run it:
tf.enable_eager_execution()

# Download the official implementation
# ! pip install -q requests
# ! git clone --depth 1 https://github.com/tensorflow/models

# Add the root directory of the repository to your Python path:
models_path = os.path.join(os.getcwd(), 'models')

sys.path.append(models_path)

# Download the dataset
from official.wide_deep import census_dataset
from official.wide_deep import census_main

census_dataset.download("/tmp/census_data/")

# Command line usage
#export PYTHONPATH=${PYTHONPATH}:"$(pwd)/models"
#running from python you need to set the `os.environ` or the subprocess will not see the directory.

if "PYTHONPATH" in os.environ:
  os.environ['PYTHONPATH'] += os.pathsep + models_path
else:
  os.environ['PYTHONPATH'] = models_path

# Use --help to see what command line options are available:
# !python -m official.wide_deep.census_main --help

# Now run the model.
# !python -m official.wide_deep.census_main --model_type=wide --train_epochs=2

# Read the U.S. Census data
# !ls  /tmp/census_data/

train_file = "/tmp/census_data/adult.data"
test_file = "/tmp/census_data/adult.test"

import pandas

train_df = pandas.read_csv(train_file, header = None, names = census_dataset._CSV_COLUMNS)
test_df = pandas.read_csv(test_file, header = None, names = census_dataset._CSV_COLUMNS)

print(train_df.head())

# Converting Data into Tensors
def easy_input_function(df, label_key, num_epochs, shuffle, batch_size):
  label = df[label_key]
  ds = tf.data.Dataset.from_tensor_slices((dict(df),label))

  if shuffle:
    ds = ds.shuffle(10000)

  ds = ds.batch(batch_size).repeat(num_epochs)

  return ds

ds = easy_input_function(train_df, label_key='income_bracket', num_epochs=5, shuffle=True, batch_size=10)

for feature_batch, label_batch in ds.take(1):
  print('Some feature keys:', list(feature_batch.keys())[:5])
  print()
  print('A batch of Ages  :', feature_batch['age'])
  print()
  print('A batch of Labels:', label_batch )

# Larger datasets should be streamed from disk.
import inspect
print(inspect.getsource(census_dataset.input_fn))

# This input_fn returns equivalent output:
ds = census_dataset.input_fn(train_file, num_epochs=5, shuffle=True, batch_size=10)

for feature_batch, label_batch in ds.take(1):
  print('Feature keys:', list(feature_batch.keys())[:5])
  print()
  print('Age batch   :', feature_batch['age'])
  print()
  print('Label batch :', label_batch )




