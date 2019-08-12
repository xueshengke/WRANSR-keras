from __future__ import print_function
import hdf5
import os
import argparse
import matplotlib.pyplot as plt
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, TensorBoard, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from utils import lr_schedule, PSNR, PSNR_Loss, SSIM, SSIM_Loss, useGPU
from models import vdsr, wransr

#-----------------------------------------------------------------------------------------------------------------------
# parser = argparse.ArgumentParser()
# parser.add_argument('-D', '--depth', default=8, dest='depth', type=int, nargs=1,
#                     help='the depth of VDSR network')
# parser.add_argument('-B', '--batch_size', default=64, dest='batch_size', type=int, nargs=1,
#                     help='batch size is the number of samples fed into the network for training each time')
# parser.add_argument('-E', '--epochs', default=200, dest='epochs', type=int, nargs=1,
#                     help='dataset will be used for training "epochs" times')
# parser.add_argument('-S', '--scale', default=4, dest='scale', type=int, nargs=1,
#                     help='the scale factor of trained VDSR network')
# parser.add_argument('-A', '--argument', default=False, dest='data_argument', type=bool, nargs=1,
#                     help='True -> use data argumentation; False - > Do not use it')
# parser.add_argument('--gpu', default='0,1', dest='gpu_id', type=str, nargs=1,
#                     help='Use GPU, for example, --gpu 0,1,2...')
# parser.add_argument('--training_path', default=os.path.join(os.getcwd(), 'data'), dest='training_path', type=str,
#                     nargs=1, help='training data and label path, should contains train/ and test/ subfolder')
# option = parser.parse_args()

# use GPU if available
useGPU('0,1')

# training parameters
scale = 4
depth = 8
ratio = 4
width = 64
alpha = 0.1
batch_size = 64
epochs = 200
data_augment = False
training_path = '/ext/xueshengke/caffe-1.0/examples/waveletCASR/data/train/wavelet_small_x4_1.h5'
model_name = 'WRANSR-%d_x%d' % (depth, scale)

# load DIV2K data (.h5 files from Matlab)
train_data, train_label, test_data, test_label = hdf5.load_data(training_path)
# h5 = h5py.File('/ext/xueshengke/caffe-1.0/examples/waveletCASR/data/train/wavelet_small_x4_1.h5', 'r')

# create WRANSR network model
input_shape = train_data[0].shape[1:]
model = wransr.wran_net(input_shape, depth, ratio, width, alpha)

# sgd = SGD(lr=1e-4, momentum=0.9, decay=1e-4, nesterov=False)
adam = Adam(lr=lr_schedule(0), beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)
model.compile(loss='mean_absolute_error', optimizer=adam, metrics=[PSNR_Loss, SSIM_Loss])
model.summary()
print(model_name)

# prepare model model saving directory
ck_dir = os.path.join(os.getcwd(), 'checkpoints')
ck_name = 'wransr%d_epoch{epoch:03d}_psnr{PSNR_Loss:02.4f}.h5' % depth
if not os.path.exists(ck_dir):
    os.makedirs(ck_dir)
ck_file = os.path.join(ck_dir, ck_name)
# prepare callbacks for model saving and for learning rate adjustment
checkpoint = ModelCheckpoint(filepath=ck_file, monitor='PSNR_Loss', verbose=1, save_best_only=True, mode='max',
                             save_weights_only='False', period=1)
# if os.path.exists(ck_file):
#     model.load_weights(ck_file)
#     print('checkpoint', ck_file, 'loaded')

# prepare save information for tensorboard
log_dir = os.path.join(os.getcwd(), 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_grads=True, write_images=True)

# early stopping policy
earlystop = EarlyStopping(monitor='PSNR_Loss', patience=5, verbose=1, mode='max')

## learning rate policy
lr_policy = LearningRateScheduler(lr_schedule)
lr_reduce = ReduceLROnPlateau(monitor='PSNR_Loss', factor=0.1, patience=3, mode='max', cooldown=0, min_lr=1e-8)

callbacks = [checkpoint,
             tensorboard,
             # earlystop,
             lr_policy,
             # lr_reduce
             ]


# start training, with or without data augmentation.
if not data_augment:
    print('Not using data augmentation.')
    history = model.fit(train_data, train_label, batch_size=batch_size, epochs=epochs,
              validation_data=(test_data, test_label), shuffle='batch', callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # this will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(train_data)

    # fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(datagen.flow(train_data, train_label, batch_size=batch_size),
                                  validation_data=(test_data, test_label),
                                  epochs=epochs, verbose=1, workers=4,
                                  callbacks=callbacks)

# evaluate trained model
scores = model.evaluate(test_data, test_label, batch_size=batch_size, verbose=1)
print('Test loss: %.8f' % scores[0], ',\tPSNR: %.4f' % scores[1])
# print('Test SSIM: %.4f' % scores[2])

# save model
print('Training complete, saving model now ...')
model.save_weights('wransr_x%d_psnr_%.4f.h5' % (scale, scores[1]))
print('Done, wransr_x%d_psnr_%.4f.h5 has been saved' % (scale, scores[1]))
del model

# save training details
# print(history.history.keys())
if not os.path.exists('figures'):
    os.mkdir('figures')
figure_dir = os.path.join('figures', 'wransr_x%d_psnr_%.4f' % (scale, scores[1]))
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)
with open(os.path.join(figure_dir, 'log.txt'), mode='w') as f:
    f.write(str(history.history))

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, 'Loss_vs_epoch.png'))
plt.show(block=False)
plt.pause(0.2)

plt.figure()
plt.plot(history.history['PSNR_Loss'])
plt.plot(history.history['val_PSNR_Loss'])
plt.title('PSNR vs epoch')
plt.ylabel('PSNR')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, 'PSNR_vs_epoch.png'))
plt.show(block=False)
plt.pause(0.2)
