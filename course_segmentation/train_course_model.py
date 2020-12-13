from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from models import *
from losses import *
from train_generator import *

import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MODEL_NAME = 'unet_course_1312'

MODEL_FILE = './checkpoints/' + MODEL_NAME + '.json'
MODEL_WEGHT = './checkpoints/' + MODEL_NAME + '.h5'
LOAD_WEIGHT = False
TRAINED_WEIGHT = './checkpoints/unet_course_1312.h5'

DATA_FOLDER = '/media/buikhoi/TrainingDat/course_segmentation/'
TRAIN_FOLDER = DATA_FOLDER + 'train/'
VAL_FOLDER = DATA_FOLDER + 'val/'

# train_model = make_fcn_model(128, 128)
train_model = unet_model()
train_model.summary()

with open(MODEL_FILE, 'w') as model_file:
    model_file.write(train_model.to_json())

if LOAD_WEIGHT:
    train_model.load_weights(TRAINED_WEIGHT)
    print('Loaded weights')

train_gen = NumpyDataGenerator(
    TRAIN_FOLDER,
    64, (128, 128),
    True
)

val_gen = NumpyDataGenerator(
    VAL_FOLDER,
    64, (128, 128),
    False
)

train_model.compile(Adam(1e-4), loss=rv_iou)
reducelr = ReduceLROnPlateau('val_loss', 0.5, 5, verbose = 1)
checkpoint = ModelCheckpoint(MODEL_WEGHT, save_best_only = True, verbose = 1)

history = train_model.fit_generator(
    train_gen, 
    train_gen.__len__(), 
    100, 
    callbacks = [
        reducelr, 
        checkpoint
    ], 
    validation_data=val_gen
)
plt.plot(history.history['val_loss'])
plt.show()
