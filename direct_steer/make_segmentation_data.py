import os
import cv2
import numpy as np
from shutil import rmtree
from keras.models import model_from_json

IMAGE_FOLDER = './images_data/'
TARGET_FOLDER = './segmented_data/'

COURSE_MODEL = './models/models/unet_course.json'
COURSE_MODEL_WEIGHT = './models/weights/unet_course.h5'

BATCH_SIZE = 10

def get_all_images(folder):
    results = []
    for f in os.listdir(folder):
        if os.path.isdir(folder + f):
            results.extend(get_all_images(folder + f + '/'))
        else:
            if any([e in f for e in ['png', 'jpg']]):
                results.append(folder + f)
    return results

if __name__ == '__main__':
    # get all images
    images = get_all_images(IMAGE_FOLDER)
    
    # clear destination folder
    if os.path.exists(TARGET_FOLDER):
        rmtree(TARGET_FOLDER)
    os.mkdir(TARGET_FOLDER)

    # load course model
    with open(COURSE_MODEL, 'r') as model_file:
        course_model = model_from_json(model_file.read())
    course_model.load_weights(COURSE_MODEL_WEIGHT)
    course_model.summary()

    print('Processing on {} images'.format(len(images)))

    # make image batch and predict
    idx = 0
    image_batch = []
    image_names = []
    while True:
        print('Processing image {} of {}'.format(idx + 1, len(images)), end='\r')
        if len(image_batch) < BATCH_SIZE and idx < len(images):
            try:
                img = cv2.imread(images[idx])[:, :, (2, 1, 0)]
                image_names.append(images[idx].split('/')[-1])
                image_batch.append(cv2.resize(img, (128, 128)))
                idx += 1
            except IndexError:
                pass
        else:
            predictions = course_model.predict(np.array(image_batch))
            for pred, img, name in zip(predictions, image_batch, image_names):
                img = cv2.bitwise_and(img, img, mask=(pred[:, :, 0] * 255).astype('uint8'))
                cv2.imwrite(TARGET_FOLDER + name, img[:, :, (2, 1, 0)])
            image_batch = []
            image_names = []
