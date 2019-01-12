from sklearn.utils import shuffle
import numpy as np
import io
import matplotlib.pyplot as plt
import base64
import pandas as pd
import tensorflow as tf
import cv2

# Function to generate an image of any size
def GetImage(imageBytes, width, height, dpi):
    buf = io.BytesIO()

    # Need to set this switch to generate the image in memory and not output to screen
    plt.switch_backend('Agg')
    image = io.BytesIO(base64.standard_b64decode(imageBytes))
    plt.figure(figsize=(width, height), dpi=dpi)
    plt.imshow(plt.imread(image))
    plt.axis('off')
    plt.savefig(buf, format='png')
    buffer = buf.getvalue()
    buf.close()
    return buffer

# Loads the training data
def load_train(dfAllImages, imageType, image_size_x, image_size_y, classes, vehicleTypes, dpi):
    images = []
    labels = []
    cls = []
    vehicleFeatures = []
   
    for index, row in dfAllImages.iterrows():

        # Generates grayscale image and pulls into a NumPy array that is then normalized
        image = GetImage(row[imageType], image_size_x / dpi, image_size_y / dpi, dpi) 
        image = cv2.imdecode(np.fromstring(image, dtype=np.uint8), 0) # <-- 0=Grayscale
        image = image.astype(np.float32)
        image = np.multiply(image, 1.0 / 255.0)
        image = np.resize(image, (image_size_y,image_size_x,1))
        images.append(image)

        # One-hot encoding of the industry
        label = np.zeros(len(classes))
        oneHotIndex = classes.index(row["NormalizedIndustry"])
        label[oneHotIndex] = 1.0
        labels.append(label)

        # One-hot encoding of the vehicle Type
        vehicleFeature = np.zeros(len(vehicleTypes))
        oneHotIndex = vehicleTypes.index(row["NormalizedVehicleType"])
        vehicleFeature[oneHotIndex] = 1.0
        vehicleFeatures.append(vehicleFeature)
        cls.append(row["NormalizedIndustry"])
    images = np.array(images)
    labels = np.array(labels)
    vehicleFeatures = np.array(vehicleFeatures)
    cls = np.array(cls)

    return images, labels, cls, vehicleFeatures


class DataSet(object):

  def __init__(self, images, labels, cls, vehicleFeature):
    self._num_examples = images.shape[0]

    self._images = images
    self._labels = labels
    self._cls = cls
    self._vehicleFeature = vehicleFeature
    self._epochs_done = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def cls(self):
    return self._cls
  
  @property
  def vehicleFeature(self):
    return self._vehicleFeature

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_done(self):
    return self._epochs_done

  # Function to get the next batch of images
  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # After each epoch we update this
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end], self._cls[start:end], self._vehicleFeature[start:end]

# Function to read in the training and validation datasets
def read_train_sets(dfAllImages, imageType, image_size_x, image_size_y, classes, vehicleTypes, validation_size, dpi):
  class DataSets(object):
    pass
  data_sets = DataSets()

  images, labels, cls, vehicleFeature = load_train(dfAllImages, imageType, image_size_x, image_size_y, classes, vehicleTypes, dpi)
  
  # Shufflles the data to obtain a different order each time
  images, labels, cls, vehicleFeature = shuffle(images, labels, cls, vehicleFeature)
  
  if isinstance(validation_size, float):
    validation_size = int(validation_size * images.shape[0])

  # Sets validation dataset
  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]
  validation_cls = cls[:validation_size]
  validation_vehicleFeature = vehicleFeature[:validation_size]

  # Sets training dataset
  train_images = images[validation_size:]
  train_labels = labels[validation_size:]
  train_cls = cls[validation_size:]
  train_vehicleFeature = cls[validation_size:]

  data_sets.train = DataSet(train_images, train_labels, train_cls, vehicleFeature)
  data_sets.valid = DataSet(validation_images, validation_labels, validation_cls, vehicleFeature)

  return data_sets