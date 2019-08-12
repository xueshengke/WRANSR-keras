from __future__ import print_function
import tensorflow as tf
import keras
from keras import backend as K
import numpy as np
import os
import keras.backend.tensorflow_backend as KTF

def useGPU(gpu_id):
    if not gpu_id:
        print('Use CPU')
    else:
        print('Use GPU')
    # only use limited GPU memory
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    KTF.set_session(sess)


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 80:
        lr *= 0.5e-3
    elif epoch > 40:
        lr *= 1e-3
    elif epoch > 20:
        lr *= 1e-2
    elif epoch > 10:
        lr *= 1e-1
    print('Current learning rate: ', lr)
    return lr

def tf_log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def PSNR_Loss(y_true, y_pred, max_pixel = 1.0):
    # assert y_true.shape == y_pred.shape, 'Cannot compute PNSR if two input shapes are not same: %s and %s' % (str(
    #     y_true.shape), str(y_pred.shape))
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))

def PSNR(y_true, y_pred, max_pixel = 1.0):
    assert y_true.shape == y_pred.shape, 'Cannot compute PNSR if two input shapes are not same: %s and %s' % (str(
        y_true.shape), str(y_pred.shape))
    return 10.0 * np.log10((max_pixel ** 2) / (np.mean(np.square(y_pred - y_true))))

def SSIM_Loss(y_true, y_pred):
    # assert y_true.shape == y_pred.shape, 'Cannot compute PNSR if two input shapes are not same: %s and %s' % (str(
    #     y_true.shape), str(y_pred.shape))
    u_true = K.mean(y_true)
    u_pred = K.mean(y_pred)
    var_true = K.var(y_true)
    var_pred = K.var(y_pred)
    std_true = K.sqrt(var_true)
    std_pred = K.sqrt(var_pred)
    c1 = K.square(0.01*7)
    c2 = K.square(0.03*7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom

def SSIM(y_true, y_pred):
    assert y_true.shape == y_pred.shape, 'Cannot compute PNSR if two input shapes are not same: %s and %s' % (str(
        y_true.shape), str(y_pred.shape))
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01*7)
    c2 = np.square(0.03*7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom

def modcrop(x, scale=1):
    w, h = x.shape[0], x.shape[1]
    w -= int(w % scale)
    h -= int(h % scale)
    if len(x.shape) == 2:
        return x[0:w, 0:h]
    elif len(x.shape) == 3:
        return x[0:w, 0:h, :]
    elif len(x.shape) == 4:
        return x[0:w, 0:h, :, :]

