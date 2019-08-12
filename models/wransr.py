from keras.models import Model
from keras.layers import Activation, Conv2D, LeakyReLU, Input, add, Concatenate
from models import attention_module

def inception_module(model, ratio=4, width=64, alpha=0.1):
    conv_1 = Conv2D(width/ratio, (1, 1), padding='same', kernel_initializer='he_normal')(model)
    conv_1 = LeakyReLU(alpha)(conv_1)

    conv_2 = Conv2D(width/ratio, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    conv_2 = LeakyReLU(alpha)(conv_2)

    conv_3 = Conv2D(width/ratio, (5, 5), padding='same', kernel_initializer='he_normal')(model)
    conv_3 = LeakyReLU(alpha)(conv_3)

    conv_4 = Conv2D(width/ratio, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    conv_4 = LeakyReLU(alpha)(conv_4)

    conv_4 = Conv2D(width/ratio, (5, 5), padding='same', kernel_initializer='he_normal')(conv_4)
    conv_4 = LeakyReLU(alpha)(conv_4)

    model = Concatenate([conv_1, conv_2, conv_3, conv_4], axis=-1)
    model = Conv2D(width, (1, 1), padding='same', kernel_initializer='he_normal')(model)
    model = LeakyReLU(alpha)(model)

    return model

def wran_net(input_shape=(None, None, 4), depth=8, ratio=4, width=64, alpha=0.1):
    input_img = Input(shape=input_shape)

    # feature extraction
    model = Conv2D(width, (5, 5), padding='same', kernel_initializer='he_normal')(input_img)
    model = LeakyReLU(alpha)(model)

    # Inception structure
    # block_in = model
    # model = inception_module(model, ratio)
    # model = attention_module(model, 'cbam_block')
    # model = add([block_in, model])

    for i in range(depth):
        block_in = model
        model = inception_module(model, ratio, width, alpha)
        model = attention_module(model, 'cbam_block')
        model = add([block_in, model])

    model = Conv2D(width, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = LeakyReLU(alpha)(model)
    model = Conv2D(4, (3, 3), padding='same', kernel_initializer='he_normal')(model)

    output_img = model    # My label is already the residual images, 'add' operation is not required

    model = Model(input_img, output_img)
    return model