import argparse
import json
import os
from . import model
import tensorflow as tf

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # Input Arguments
  parser.add_argument(
      '--train_batch_size',
      help='Batch size for training steps',
      type=int,
      default=100)
  parser.add_argument(
      '--validation_size',
      help='Validation percentage of total data',
      type=float,
      default=0.2)
  parser.add_argument(
      '--num_records',
      help='Number of records to pull in from BQ',
      type=int,
      default=2000)
  parser.add_argument(
      '--learning_rate',
      help='Initial learning rate for training',
      type=float,
      default=0.00001)
  parser.add_argument(
      '--train_steps',
      help="""\
      Steps to run the training job for. A step is one batch-size,\
      """,
      type=int,
      default=0)
  parser.add_argument(
      '--output_dir',
      help='GCS location to write checkpoints and export models',
      required=True,
      default='gs://geotab-bi-eu/industry-classifier/trainer/output_dir2')

  #generate list of model functions to print in help message
  model_names = [name.replace('_model','') \
                   for name in dir(model) \
                     if name.endswith('_model')]

  #  Hyperparameters used by CNN
  parser.add_argument(
      '--ksize1',
      help='kernel size of first layer for CNN',
      type=int,
      default=3)
  parser.add_argument(
      '--ksize2',
      help='kernel size of second layer for CNN',
      type=int,
      default=3)
  parser.add_argument(
      '--ksize3',
      help='kernel size of third layer for CNN',
      type=int,
      default=3)
  parser.add_argument(
      '--nfil1',
      help='number of filters in first layer for CNN',
      type=int,
      default=32)
  parser.add_argument(
      '--nfil2',
      help='number of filters in second layer for CNN',
      type=int,
      default=32)
  parser.add_argument(
      '--nfil3',
      help='number of filters in second layer for CNN',
      type=int,
      default=64)
  parser.add_argument(
      '--dprob',
      help='dropout probability for CNN',
      type=float,
      default=0.25)
  parser.add_argument(
      '--fc_layer_size',
      help='fully connected layer size',
      type=int,
      default=128)
  parser.add_argument(
      '--dropout_rate',
      help='dropout rate for dense layers',
      type=float,
      default=0.4)
  parser.add_argument(
      '--conv_stride',
      help='convolution stride',
      type=int,
      default=1)
  parser.add_argument(
      '--max_pool_ksize',
      help='max pool filter size',
      type=int,
      default=2)
  parser.add_argument(
      '--max_pool_stride',
      help='max pool stride',
      type=int,
      default=2)
  parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True)

  args = parser.parse_args()
  hparams = args.__dict__
  
  hparams.pop('job_dir')
  output_dir = hparams.pop('output_dir')

  # Append trial_id to path so tuning jobs don't overwrite each other
  output_dir = os.path.join(
      output_dir,
      json.loads(
          os.environ.get('TF_CONFIG', '{}')
      ).get('task', {}).get('trial', '')
  )

  # calculate train_steps if not provided
  if hparams['train_steps'] < 1:
     hparams['train_steps'] = (10000 * 20) // hparams['train_batch_size']
     print("Training for {} steps".format(hparams['train_steps']))
  
  # Run the training job from model.py
  model.train_and_evaluate(output_dir, hparams)