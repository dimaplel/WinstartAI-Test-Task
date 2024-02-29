from tensorflow.keras import models, layers


# Helper function for encoding and decoding blocks
def conv_block(input, filters, drop_rate):
    x = layers.Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(input)
    x = layers.Dropout(drop_rate)(x)
    x = layers.Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    return x


# Encoding block for U-Net architecture
def encoding_block(input, filters, drop_rate):
    x = conv_block(input, filters, drop_rate)
    return x, layers.MaxPooling2D((2, 2))(x)


# Decoding block for U-Net architecture
def decoder_block(input, processed, filters, drop_rate):
    x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(input)
    x = layers.Concatenate()([x, processed])
    return conv_block(x, filters, drop_rate)


# Function which builds model based on U-Net architecture
def unet(pretrained_weights=None, input_size=(256, 256, 3)):
    inputs = layers.Input(input_size)

    s1, p1 = encoding_block(inputs, 16, 0.1)
    s2, p2 = encoding_block(p1, 32, 0.1)
    s3, p3 = encoding_block(p2, 64, 0.2)
    s4, p4 = encoding_block(p3, 128, 0.2)

    b = conv_block(p4, 256, 0.3)

    d1 = decoder_block(b, s4, 128, 0.2)
    d2 = decoder_block(d1, s3, 64, 0.2)
    d3 = decoder_block(d2, s2, 32, 0.1)
    d4 = decoder_block(d3, s1, 16, 0.1)

    outputs = layers.Conv2D(1, (1, 1), padding="same", activation='sigmoid')(d4)

    model = models.Model(inputs=[inputs], outputs=[outputs], name="U-Net")

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
