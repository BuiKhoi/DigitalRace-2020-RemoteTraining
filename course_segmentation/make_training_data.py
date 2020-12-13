import os
import time
import json
import numpy as np
import cv2
from shutil import rmtree
import albumentations as A
from threading import Thread
from generate_images import get_files

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.7),
    A.RandomGamma(p=0.7),
    A.CLAHE(p=0.5),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1),
])

def millis():
    return int(round(time.time() * 1000))

def clean_folders(save_folder):
    for f in ['train/', 'val/']:
        if os.path.exists(save_folder + f):
            rmtree(save_folder + f)
        os.mkdir(save_folder + f)

def save_npy(image, annotation, dir):
    if image.max() > 1:
        image = (image/255).astype(np.float16)
        if image.shape != (240, 320, 3):
            print(image.shape)
    annotation = annotation.astype(np.float16)
    decide = np.random.random()
    if decide < 0.2:
        np.savez_compressed(dir + 'val/' + str(millis()), img = image, anno = annotation)
    else:
        np.savez_compressed(dir + 'train/' + str(millis()), img = image, anno = annotation)

def process_image(img_file, save_dir, classes):
    file_name = img_file[0:-5]
    img = cv2.imread(file_name + '.jpg')[:, :, (2, 1, 0)]
    anno = None
    with open(file_name + '.json', 'r') as read_file:
        anno = json.loads(read_file.read())
    if classes == 1:
        label = np.zeros((240, 320), np.int8)
    for s in anno['shapes']:
        try:
            pts = []
            for p in s['points']:
                pts.append(np.array([np.array(p)]))
            pts = np.array(pts)
            temp = np.zeros((240, 320))
            cv2.drawContours(temp, [pts.astype('int')], 0, 1, -1)
            label = temp.copy()
        except ValueError:
            continue
    label = np.clip(label, 0, 1)
    save_npy(img, label, save_dir)

    for i in range(2):
        transformed = transform(image=img)
        save_npy(transformed["image"].astype(np.uint8), label, save_dir)

def parse_annotations(images, save_dir, classes=6):
    processing_threads = []
    max_threads = 20

    for idx, img_file in enumerate(images):
        print('Processing file {} of {}'.format(idx, len(images)), end='\r')
        processing_threads.append(Thread(target=process_image, args=[img_file, save_dir, classes]))
        processing_threads[-1].start()
        time.sleep(0.001)
        if len(processing_threads) == max_threads:
            while all([t.is_alive() for t in processing_threads]):
                time.sleep(0.001)

            for t in reversed(processing_threads):
                if not t.is_alive():
                    processing_threads.remove(t)

def check_processed_images(train_folder):
    for saved_dir in [train_folder + f for f in ['train/', 'val/']]:
        error_count = 0
        saved_images = [saved_dir + f for f in os.listdir(saved_dir)]

        for idx, img in enumerate(saved_images):
            print('Counting image {}'.format(idx), end='\r')
            try:
                np.load(img)['img']
            except:
                error_count += 1
                os.remove(img)
        print('{} error(s) on a total of {} images'.format(error_count, len(saved_images)))

if __name__ == '__main__':
    images_folder = './raw_images/'
    generated_file = './generated_images.txt'
    save_folder = '/media/buikhoi/TrainingDat/course_segmentation/'
    print('Cleaning target folder')
    clean_folders(save_folder)

    print('Getting all images')
    images = get_files(images_folder, generated_file)

    print('Parsing annotations')
    parse_annotations(images, save_folder, 1)

    print('\nDouble checking data')
    check_processed_images(save_folder)

    print('Ok done')