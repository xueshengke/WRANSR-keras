from keras.models import Model
from keras.layers import Activation
from keras.layers import Conv2D, Input, add

def vdsr_net(input_shape=(None, None, 4), depth=20):
    input_img = Input(shape=input_shape)

    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
    model = Activation('relu')(model)
    for i in range(depth-2):
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = Activation('relu')(model)
    model = Conv2D(4, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    res_img = model

    # output_img = add([res_img, input_img])
    output_img = res_img    # My label is already the residual, add is not required

    model = Model(input_img, output_img)

    return model