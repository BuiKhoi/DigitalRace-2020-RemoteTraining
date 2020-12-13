from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, UpSampling2D, Input, Lambda, Conv2DTranspose, LeakyReLU, concatenate

def unet_model(input_size = (128, 128, 3), n_classes = 1):
  inputs = Input(input_size)
  conv1 = Conv2D(16, 3, padding='same', activation = 'relu', kernel_initializer='he_normal')(inputs)
  conv1 = Conv2D(32, 3, padding='same', activation = 'relu', kernel_initializer='he_normal')(conv1)
  conv1 = Dropout(0.5)(conv1)
  pool1 = MaxPool2D((4, 4))(conv1)

  # conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
  # conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
  # conv2 = Dropout(0.5)(conv2)
  # pool2 = MaxPool2D((4, 4))(conv2)

  conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
  conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
  drop3 = Dropout(0.5)(conv3)
  
  # up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D((4, 4))(drop3))
  # merge8 = concatenate([conv2, up8], axis=3)
  # conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
  # conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

  up9 = Conv2D(64, 2, padding='same', activation = 'relu', kernel_initializer='he_normal')(UpSampling2D((4, 4))(drop3))
  merge9 = concatenate([conv1, up9], axis=3)
  conv9 = Conv2D(32, 3, padding='same', activation = 'relu', kernel_initializer='he_normal')(merge9)
  conv9 = Conv2D(32, 3, padding='same', activation = 'relu', kernel_initializer='he_normal')(conv9)
  conv9 = Conv2D(16, 3, padding='same', activation = 'relu', kernel_initializer='he_normal')(conv9)
  conv10 = Conv2D(n_classes, 1, activation='sigmoid')(conv9)

  model = Model(inputs=inputs, outputs=conv10)
  return model

def make_fcn_model(IMG_HEIGHT, IMG_WIDTH):
    b = 4
    i = Input((IMG_HEIGHT, IMG_WIDTH, 3))
    # s = Lambda(lambda x: preprocess_input(x)) (i)
    c1 = Conv2D(2**b, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (i)
    c1 = Conv2D(2**b, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
    c1 = Conv2D(2**b, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPool2D((8, 8)) (c1)

    # c2 = Conv2D(2**(b+1), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
    # c2 = Conv2D(2**(b+1), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
    # c2 = Dropout(0.1) (c2)
    # c2 = Conv2D(2**(b+1), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
    # p2 = MaxPool2D((2, 2)) (c2)

    c3 = Conv2D(2**(b+2), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
    c3 = Conv2D(2**(b+2), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(2**(b+2), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPool2D((8, 8)) (c3)

    # c4 = Conv2D(2**(b+3), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
    # c4 = Conv2D(2**(b+3), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
    # c4 = Dropout(0.2) (c4)
    # c4 = Conv2D(2**(b+3), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
    # p4 = MaxPool2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(2**(b+4), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
    c5 = Conv2D(2**(b+4), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(2**(b+4), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)
    c5 = Conv2D(2**(b+4), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)

    # u6 = Conv2DTranspose(2**(b+3), (2, 2), strides=(2, 2), padding='same') (c5)
    # u6 = concatenate([u6, c4])
    # c6 = Conv2D(2**(b+3), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
    # c6 = Dropout(0.2) (c6)
    # c6 = Conv2D(2**(b+3), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
    # c6 = Conv2D(2**(b+3), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(2**(b+2), (2, 2), strides=(8, 8), padding='same') (c5)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(2**(b+2), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Conv2D(2**(b+2), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(2**(b+2), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)

    # u8 = Conv2DTranspose(2**(b+1), (2, 2), strides=(2, 2), padding='same') (c7)
    # u8 = concatenate([u8, c2])
    # c8 = Conv2D(2**(b+1), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
    # c8 = Conv2D(2**(b+1), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
    # c8 = Dropout(0.1) (c8)
    # c8 = Conv2D(2**(b+1), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(2**b, (2, 2), strides=(8, 8), padding='same') (c7)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(2**b, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Conv2D(2**b, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)

    o = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=i, outputs=o)
    return model