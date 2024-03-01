import os
import random

from tensorflow import keras
from matplotlib import pyplot as plt

from config import *
from utils.generators import create_pred_gen
from utils.metrics import DiceLoss, dice_coef

n_images = 4

all_test_images = os.listdir(test_dir)
# Randomly selecting test images
test_imgs = random.sample(all_test_images, min(n_images, len(all_test_images)))
# Loading model
model = keras.models.load_model(model_path, custom_objects={'DiceLoss': DiceLoss, 'dice_coef': dice_coef})

# Plotting images and their masks
rows = 1
columns = 2
for i in range(len(test_imgs)):
    img, pred = create_pred_gen(test_dir, test_imgs[i], model)
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(rows, columns, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Image")
    fig.add_subplot(rows, columns, 2)
    plt.imshow(pred, interpolation=None)
    plt.axis('off')
    plt.title("Prediction Mask")
    plt.show()
