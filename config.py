# Directories paths
base_dir = 'dataset/'
train_dir = base_dir + 'train_v2/'
test_dir = base_dir + 'test_v2/'
model_path = "models/{}_model_v1.h5".format('unet')
weights_path = "models/{}_weights_v1.h5".format('unet')

# Model parametes
batch_size = 48
img_scale = (3, 3)
validation_batch = 900
steps = 10
epochs = 99

# ImageDataGenerator parameters
dg_params = {
    "rotation_range": 30,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "shear_range": 0.01,
    "zoom_range": [0.9, 1.25],
    "horizontal_flip": True,
    "vertical_flip": True,
    "fill_mode": "reflect",
    "data_format": "channels_last"
}
