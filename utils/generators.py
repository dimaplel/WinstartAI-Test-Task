import os
import cv2
import numpy as np

from skimage.io import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import expand_dims

from config import dg_params, batch_size, img_scale, train_dir
from utils.utils import decode_image

# Create image data generators with parameters from config
image_gen = ImageDataGenerator(**dg_params)
label_gen = ImageDataGenerator(**dg_params)


# Function for creating generator with batch size from dataframe with ImageId and EncodedPixels
def create_gen(df, batch_size=batch_size):
    batches = list(df.groupby('ImageId'))
    img_batch = []
    masks_batch = []
    while True:
        np.random.shuffle(batches)
        for img_id, masks in batches:
            img_path = os.path.join(train_dir, img_id)
            img = imread(img_path)
            mask = np.expand_dims(decode_image(masks['EncodedPixels'].values), -1)

            if img_scale is not None:
                img = img[::img_scale[0], ::img_scale[1]]
                mask = mask[::img_scale[0], ::img_scale[1]]

            img_batch.append(img)
            masks_batch.append(mask)
            if len(img_batch) >= batch_size:
                yield np.stack(img_batch, 0) / 255., np.stack(masks_batch, 0).astype(np.float32)
                img_batch, masks_batch = [], []


# Creating augmented image data generator from basic generator
def create_aug_gen(gen, seed=None):
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    for im, masks in gen:
        seed = np.random.choice(range(9999))
        g_x = image_gen.flow(im,
                             batch_size=im.shape[0],
                             seed=seed,
                             shuffle=True)
        g_y = label_gen.flow(masks,
                             batch_size=im.shape[0],
                             seed=seed,
                             shuffle=True)

        yield next(g_x), next(g_y)


# Generator for predictions
def create_pred_gen(test_dir, img, model):
    img_path = os.path.join(test_dir,img)
    img = imread(img_path)
    img = expand_dims(img, axis=0)
    pred = model.predict(img)
    pred = np.squeeze(pred, axis=0)
    return cv2.imread(img_path), pred
