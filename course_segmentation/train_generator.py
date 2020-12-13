import os
import cv2
import numpy as np
from tensorflow import keras

class NumpyDataGenerator(keras.utils.Sequence):
  'Generates data for Keras'
  def __init__(self, images_link, batch_size=32, dim=(240, 320), shuffle=False):
    self.dim = dim
    self.batch_size = batch_size
    self.images_link = images_link
    self.shuffle = shuffle

    self.images = os.listdir(self.images_link)
    print('Generator on {} images'.format(len(self.images)))
    self.on_epoch_end()

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.images) / self.batch_size))

  def __getitem__(self, index):
    'Generate one batch of data'
    #print('Index: {}'.format(index))
    # Generate indexes of the batch
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    list_images_temp = [self.images[k] for k in indexes]

    # Generate data
    # X, y = self.__data_generation(list_images_temp)
    X, y = self.__data_generation(list_images_temp)

    return X, y

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.images))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def __data_generation(self, list_images_temp):
    '''
    Generate data with the specified batch size
    '''
    # Initialization
    X = np.empty((self.batch_size, *self.dim, 3), dtype=np.float32)
    y = np.empty((self.batch_size, *self.dim, 1), dtype=np.float16)

    for i, image in enumerate(list_images_temp):
      temp = np.load(self.images_link + image)

      X[i,] = cv2.resize(temp['img'].astype(np.float32), (128, 128))
      y[i, :, :, 0] = cv2.resize(temp['anno'].astype(np.float32), (128, 128))
    #   X[i,] = temp['img']
    #   y[i,] = temp['anno']

    return X, y