import os
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split

from utils.utils import undersample
from utils.generators import create_gen, create_aug_gen
from utils.metrics import DiceBCELoss, dice_coef
from utils.model_build import unet
from config import base_dir, train_dir, test_dir, batch_size, validation_batch, model_path, steps, epochs


# Listing training and testing files
train_files = os.listdir(train_dir)
test_files = os.listdir(test_dir)

# Data preparation
train_masks = pd.read_csv(os.path.join(base_dir, 'train_ship_segmentations_v2.csv'))
# Creating column for ships availability
train_masks['has_ships'] = train_masks['EncodedPixels'].map(lambda x: 1 if isinstance(x, str) else 0)

# Creating a dataframe with grouping by ImageId to combine all masks, counting amount of ships for each image
unique_img_df = train_masks.groupby('ImageId').agg({'has_ships': 'sum'}).reset_index()
unique_img_df.rename(columns={"has_ships": "ships_count"}, inplace=True)
# Creating helper column for ship availability
unique_img_df['has_ships'] = unique_img_df['ships_count'].map(lambda x: 1.0 if x > 0 else 0.0)
train_masks.drop(['has_ships'], axis=1, inplace=True)

# Grouping unique images' ship counts together for balancing, applying undersample function
unique_img_df['grouped_ships_count'] = unique_img_df['ships_count'].map(lambda x: (x + 1) // 2).clip(0, 7)
balanced_img_df = unique_img_df.groupby('grouped_ships_count', group_keys=False).apply(undersample)

# Splitting training dataset into training and validation subsets
train_ids, valid_ids = train_test_split(balanced_img_df,
                                        test_size=0.2,
                                        stratify=balanced_img_df['ships_count'])

train_df = pd.merge(train_masks, train_ids)
val_df = pd.merge(train_masks, valid_ids)
print("Training data was split into:")
print('training images\t\t', train_df.shape[0])
print('validation images\t', val_df.shape[0])

# Creating validation generator and getting the first batch
X_val, y_val = next(create_gen(val_df, validation_batch))
print(X_val.shape, y_val.shape)

# Creating segmentation model
model = unet()

# Checkpoint for saving model on best validation dice coefficient
checkpoint = ModelCheckpoint(model_path,
                             monitor='val_dice_coef',
                             verbose=1,
                             mode='max',
                             save_weights_only=True,
                             save_best_only=True)
# Checkpoint for reducing learning rate on not improving validation metric
reduceLR = ReduceLROnPlateau(monitor='val_dice_coef',
                             factor=0.2,
                             patience=5,
                             verbose=1,
                             mode='max',
                             min_delta=0.0001,
                             cooldown=2,
                             min_lr=1e-6)
# Early stop on 20 epochs if val metric has not improved
early = EarlyStopping(monitor='val_dice_coef',
                      mode="max",
                      patience=20)

callbacks_list = [checkpoint, early, reduceLR]

# Compiling model with Adam optimizer, DiceDCE as loss and dice coefficient as metric
model.compile(optimizer=Adam(1e-3), loss=DiceBCELoss, metrics=[dice_coef])

# Defining steps count
step_count = min(steps, train_df.shape[0] // batch_size)
# Creating training augmented generator
aug_gen = create_aug_gen(create_gen(train_df))

# Training model
loss_history = model.fit(aug_gen,
                         steps_per_epoch=steps,
                         epochs=epochs,
                         validation_data=(X_val, y_val),
                         callbacks=callbacks_list)
