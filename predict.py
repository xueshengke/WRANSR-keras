from __future__ import print_function
import os
import argparse
import pywt
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from models import vdsr, wransr
from utils import PSNR, SSIM, modcrop, useGPU
from scipy.misc import imsave, imread, imresize

# argument definition
parser = argparse.ArgumentParser()
parser.add_argument('-D', '--depth', default=20, dest='depth', type=int, nargs=1,
                    help='the depth of VDSR network')
parser.add_argument('-W', '--weights', default='', dest='weights', type=str, nargs=1,
                    help='the weights of trained VDSR network')
parser.add_argument('-S', '--scale', default=4, dest='scale', type=int, nargs=1,
                    help='the scale factor of trained VDSR network')
parser.add_argument('--gpu', default='', dest='gpu_id', type=str, nargs=1,
                    help='Use GPU, for example, --gpu 0,1,2...')
parser.add_argument('--dataset', default='/ext/xueshengke/dataset/Set5', dest='dataset', type=str, nargs=1,
                    help='dataset fot testing the trained model')
parser.add_argument('--display', default=True, dest='display', type=bool, nargs=1,
                    help='display generated images')
option = parser.parse_args()

# parse argument
scale = option.scale
depth = option.depth
weights_name = option.weights
test_dataset = option.dataset

# define our own parameters
scale = 4
depth = 20
ratio = 4
width = 64
alpha = 0.1
weights_name = '***.h5' # need to give a specific model file
test_dataset = '/ext/xueshengke/dataset/'+'Set5'
# test_dataset = '/ext/xueshengke/dataset/'+'Set14'
# test_dataset = '/ext/xueshengke/dataset/'+'BSDS100'
# test_dataset = '/ext/xueshengke/dataset/'+'Urban100'

# use GPU if available
useGPU(option.gpu_id)

## prepare directory for results
save_dir = 'results'
scale_dir = 'x%d' % scale
dataset_name = test_dataset.split('/')[-1]
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(os.path.join(save_dir, scale_dir)):
    os.mkdir(os.path.join(save_dir, scale_dir))
if not os.path.exists(os.path.join(save_dir, scale_dir, dataset_name)):
    os.mkdir(os.path.join(save_dir, scale_dir, dataset_name))
result_dir = os.path.join(save_dir, scale_dir, dataset_name)

# load pretrained model
input_shape = (None, None, 4)
model = wransr.wran_net(input_shape, depth, ratio, width, alpha)

model.load_weights(weights_name)
# model.summary()

# evaluate each image in dataset directory
image_list = []
psnr_list = []
ssim_list = []
file_dir = os.listdir(test_dataset)

for file in file_dir:
    # read image and prepare input
    image_name = file
    image_list.append(image_name)
    img = imread(os.path.join(test_dataset, image_name), mode='YCbCr')
    x = np.array(img[:,:,0])
    x = modcrop(x, scale)
    x_lr = imresize(x, 1.0/scale, 'bicubic') / 255.0
    x_bic = imresize(x_lr, 1.0*scale, 'bicubic') / 255.0
    x = x / 255.0
    # Wavelet transform
    cA, (cH, cV, cD) = pywt.dwt2(x_bic, 'haar')
    input_data = np.array([[cA, cH, cV, cD]])
    input_data = input_data.transpose([0,2,3,1])

    # predict by pretrained model
    result = model.predict(input_data, batch_size=1, verbose=1)
    result = np.squeeze(result)
    # inverse Wavelet transform
    rA, rH, rV, rD = result[:,:,0], result[:,:,1], result[:,:,2], result[:,:,3]
    x_sr = pywt.idwt2((rA, (rH, rV, rD)), 'haar')

    # compute metrics, remove border first
    psnr_val = PSNR(x[scale:-scale, scale:-scale], x_sr[scale:-scale, scale:-scale]+x_bic[scale:-scale, scale:-scale])
    ssim_val = SSIM(x[scale:-scale, scale:-scale], x_sr[scale:-scale, scale:-scale]+x_bic[scale:-scale, scale:-scale])
    psnr_list.append(psnr_val)
    ssim_list.append(ssim_val)
    print('%s\tPSNR: %.4f\tSSIM: %.4f' % (image_name, psnr_val, ssim_val))

    # display images
    if option.display:
        plt.figure()
        plt.subplot(221)
        plt.imshow(x_bic)
        plt.title('bicubic'), plt.axis('off')
        plt.subplot(222)
        plt.imshow(x_sr)
        plt.title('SR'), plt.axis('off')
        plt.subplot(223)
        plt.imshow(x_bic+x_sr)
        plt.title('bicubic+SR'), plt.axis('off')
        plt.subplot(224)
        plt.imshow(x)
        plt.title('ground truth'), plt.axis('off')
        plt.show(block=False)
        plt.pause(0.2)

    # save generated image
    imsave(os.path.join(result_dir, image_name[:-4]+'_x%d_SR.png' % scale), x_bic + x_sr)

# compute mean values
image_list.append('Mean')
psnr_list.append(np.mean(psnr_list))
ssim_list.append(np.mean(ssim_list))
print('%s\tPSNR: %.4f\tSSIM: %.4f' % (image_list[-1], psnr_list[-1], ssim_list[-1]))

# save metrics to csv file
with open(os.path.join(result_dir, 'metrics_x%d_SR.csv' % scale), mode='w') as f:
    writer = csv.writer(f)
    writer.writerow(['image', 'PSNR', 'SSIM'])
    for row in zip(image_list, psnr_list, ssim_list):
        writer.writerow(row)

print('Complete!')