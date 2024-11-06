#import libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import pickle
import time


import sys
import torch

import math
#import skimage
from skimage.feature import hog
import skimage
#ndimage
from scipy import ndimage
#import #we will use scipy.ndimage.gaussian_filter for gaussian blurring
from scipy.ndimage import gaussian_filter
#import kernal
from scipy.signal import convolve2d
#gaussian kernal
from scipy.ndimage.filters import gaussian_filter
#import PIl Image
from PIL import Image
import PIL
from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
import cv2
# if calculate_contractive_factor:
#             #we will calculate the contraction factor P =WG 
#             #we will use power method to calculate the largest eigenvalue of P=WG
#             #for power method we will call the function : power_method_for_images
#             #we will construc the argument dict args_dict for calling the function
#             #we will add input_image, denoiser, denoiser_kwargs, A_function, A_kwargs, A_function_adjoint, A_adjoint_kwargs, eta
#             args_dict_P = {'denoiser': denoiser, 'denoiser_kwargs': denoiser_kwargs, \
#                 'A_function': A_function, 'A_kwargs': A_kwargs, 'A_function_adjoint': A_function_adjoint, \
#                 'A_adjoint_kwargs': A_adjoint_kwargs, 'eta': eta}
#             #if the denoiser is DSG_NLM, then we will use the function P_operator for f in the power method function
#             #if its NLM, we will use f= D_half_P_D_half_inverse
#             if denoiser.__name__ == 'DSG_NLM':
#                 function_spectral_norm = P_operator
#                 function_spectral_norm_adjoint = P_operator_adjoint
#             elif denoiser.__name__ == 'NLM':
#                 function_spectral_norm = D_half_P_D_half_inverse
#                 function_spectral_norm_adjoint = D_half_P_D_half_inverse_adjoint
#                 #we need to add the keyword argument 'D_matrix' to the args_dict_P
#                 #we can get teh D_matrix by calling teh denoiser with an extra keyword argument 'return_D_matrix' = True
#                 #add the keyword argument 'return_D_matrix' = True to the denoiser_kwargs
#                 denoiser_kwargs['return_D_matrix'] = True
#                 #call the denoiser
#                 _, D_matrix = denoiser(denoiser_kwargs['guide_image'], **denoiser_kwargs)
#                 #drop the keyword argument 'return_D_matrix' = True from the denoiser_kwargs dictionary
#                 del denoiser_kwargs['return_D_matrix']
#                 #add the D_matrix to the args_dict_P
#                 args_dict_P['D_matrix'] = D_matrix

#             else:
#                 #assert error
#                 assert False, 'denoiser is not implemented'
#             #call the power method function
#             contractive_factor = power_method_for_images(f = function_spectral_norm, input_image = x_k, \
#                 args_dict = args_dict_P, max_iterations=power_method_max_iterations )
#             # contractive_factor = power_method_for_images_non_symmetric(functional = function_spectral_norm, functional_adjoint = function_spectral_norm_adjoint, \
#                 # image_height = x_k.shape[0], image_width = x_k.shape[1], args_dict_functional = args_dict_P, \
#                 #  args_dict_functional_adjoint = args_dict_P, max_iterations=power_method_max_iterations )   
#             # #append the contractive factor to the list
#             # contractive_factor_list.append(contractive_factor)
#             #print the contractive factor
#             print('contractive factor: ', contractive_factor)
#             #contractive factor ends ----------------------------------------------------------------------

#         # #check that the keword argumnt in dictionary 'guide_image' is present and not None
#         # assert 'guide_image' in denoiser_kwargs.keys() and denoiser_kwargs['guide_image'] is not None
#         #denoise the updat
from skimage import io
from skimage import color
from skimage import img_as_float
import skimage.measure as measure
from skimage import metrics
from skimage import img_as_ubyte
import os
import time
import argparse
import random

import matplotlib.pyplot as plt
from torchsummary import summary
#imports for dataset and dataloader
import PIL
from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
import cv2
#math
import math
#import copy
import copy
import os

from scipy.signal import medfilt2d
from scipy.ndimage import median_filter


## helper functions
#we will write a function to return a path
#the number 01 would be changed to 02, 03, 04, 05, 06, 07, 08, 09, 10, .. 14
def get_path(image_path, number):
    #number should be between 1 to 14
    #assert
    assert number >= 1 and number <= 16, "number should be between 1 to 14"
    path = image_path + str(number) + '.png'
    #check if path exists, else raise error
    if not os.path.exists(path):
        raise Exception('path does not exist')
    return path
#write a function to save the image
#input: image_array, save_directory, image_name
def save_image(image_array, save_directory, image_name):
    #create output file path, using the save_directory and image_name , os.path.join
    output_file_path = os.path.join(save_directory, image_name)
    #extension of the image is .png
    output_file_path = output_file_path + '.png'
    cv2.imwrite(output_file_path, image_array*255, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    #write a function to plot the psnr, ssim and euclidean distance against the number of iterations

def plot_graphs(quality_list_dict,  iterate_list, iterations, contractive_factor_list=None):
    psnr_list = quality_list_dict['psnr_list']
    ssim_list = quality_list_dict['ssim_list']
    mse_list = quality_list_dict['log_l2_list']
    euclidean_distance_list = quality_list_dict['log_euclidean_distance_list']

    plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.plot(iterations, psnr_list)
    plt.xlabel('iterations')
    plt.ylabel('psnr')
    plt.title('psnr')
    plt.subplot(2,2,2)
    plt.plot(iterations, ssim_list)
    plt.xlabel('iterations')
    plt.ylabel('ssim')
    plt.title('ssim')
    plt.subplot(2,2,3)
    plt.plot(iterations, mse_list)
    plt.xlabel('iterations')
    plt.ylabel('log l2 norm')
    plt.title('log residual')
    plt.subplot(2,2,4)
    plt.plot(iterations, euclidean_distance_list)
    plt.xlabel('iterations')
    plt.ylabel('log l2 norm')
    plt.title('iterate distance log')
    plt.show()
    if contractive_factor_list is not None:
        plt.figure(figsize=(10,10))
        plt.plot(iterations, contractive_factor_list)
        plt.xlabel('iterations')
        plt.ylabel('contractive factor')
        plt.title('||P||')
        plt.show()

    #plot the iterate
    plt.figure(figsize=(10,10))
    plt.plot(iterations, iterate_list)
    plt.xlabel('iterations')
    plt.ylabel('||x_k - x_inf||_2')
    plt.title('Convergence Rate')
    plt.show()


#write a function to apped the image quality metrics to a list and print them
#input: image_quality_metrics_dict, euclidean_distance, quality_list_dict, iteration
#quality_list_dict has the following keys: psnr_list, ssim_list, mse_list, euclidean_distance_list

def append_and_print_quality_metrics(image_quality_metrics_dict, log_euclidean_distance, quality_list_dict, iteration):
    quality_list_dict['psnr_list'].append(image_quality_metrics_dict['psnr'])
    quality_list_dict['ssim_list'].append(image_quality_metrics_dict['ssim'])
    quality_list_dict['log_l2_list'].append(image_quality_metrics_dict['log_l2'])
    quality_list_dict['log_euclidean_distance_list'].append(log_euclidean_distance)
    print('iteration: ', iteration, '*'*5)
    print('psnr: ', image_quality_metrics_dict['psnr'],'ssim: ', image_quality_metrics_dict['ssim'])
    # print('mse: ', image_quality_metrics_dict['mse'])
    print('log euclidean_distance: ', log_euclidean_distance)
    return quality_list_dict

def plot_iterates(quality_list_dict, iterate_list , x0, image, b, x_k, output_directory, image_number):
    #create an array of iterations to return
    iterations = np.arange(1, len(quality_list_dict['psnr_list'])+1)
    #we will get the euclidean norm of each element in the iterate list wrt the last element
    convergence_point = iterate_list[-1]
    iterate_list = iterate_list[:-1]
    iterate_list = [np.log(np.linalg.norm(x - convergence_point)) for x in iterate_list]
    # plot  psnr_list, ssim_list, mse_list, euclidean_distance_list, iterations
    plot_graphs( quality_list_dict = quality_list_dict, iterate_list=iterate_list, iterations=iterations)
    #plot the images, image, b, x_k
    #create a figure
    fig = plt.figure(figsize=(10,10))
    #add the initial image xo
    ax = fig.add_subplot(1,4,1)
    ax.imshow(x0, cmap='gray', vmin=0, vmax=1)
    ax.set_title('initial image')
    #add the original image
    ax = fig.add_subplot(1,4,2)
    ax.imshow(image, cmap='gray', vmin=0, vmax=1)
    ax.set_title('original image')
    #add the observed image
    ax = fig.add_subplot(1,4,3)
    ax.imshow(b, cmap='gray', vmin=0, vmax=1)
    ax.set_title('observed image')
    #add the reconstructed image
    ax = fig.add_subplot(1,4,4)
    ax.imshow(x_k, cmap='gray', vmin=0, vmax=1)
    ax.set_title('reconstructed image')
    plt.show()
    #close the figure
    plt.close(fig)

    #save the initial image, we will call function save_image
    save_image(image_array = x0, save_directory=output_directory, image_name=str(image_number)+'_initial')
    #save the observed image
    save_image(image_array = b, save_directory=output_directory, image_name=str(image_number)+'_observed')
    #save the reconstructed image
    save_image(image_array = x_k, save_directory=output_directory, image_name=str(image_number)+'_reconstructed')



#----------------------- INOUT IMAGE AUGUMENTATION -----------------------#
#-----------------------                           -----------------------#

#we will write a fucntion to validate and augument an image for superresolution
#as superresolution involves downsampling the image by downsample_fraction_x, downsample_fraction_y
#and then upsampling the image by upsample_fraction_x, upsample_fraction_y
#so we have only implemented downsample_fraction_x, downsample_fraction_y as 1/downsample_fraction_x, 1/downsample_fraction_y
#are integeres
#we define scale_x = 1/downsample_fraction_x, scale_y = 1/downsample_fraction_y
#so if the original image has original_image_rows, original_image_cols, we will be 
#first checking that original_image_rows is divisible by scale_x and original_image_cols is divisible by scale_y
#if yes then, we wont augument the image, but oif not
#then we will add rows and columns to the image to make it divisible by scale_x and scale_y
#we will use reflection padding to pad the image
def augument_image_superresolution(image, downsample_fraction_x, downsample_fraction_y):
    #get the shape of the image
    #get the size of the image
    original_image_rows , original_image_cols = image.shape
    #get the scale
    scale_x = 1/downsample_fraction_x
    scale_y = 1/downsample_fraction_y
    #check if scale is integer, if not assert error
    if scale_x.is_integer() and scale_y.is_integer():
        #we will convert to integer
        scale_x = int(scale_x)
        scale_y = int(scale_y)
    else:
        #raise error
        raise AssertionError("scale_x and scale_y must be integers")
    #we will first get the number of rows short of a multiple of scale_x
    if original_image_rows % scale_x == 0:
        #we wont augument the image
        rows_short = 0
    else:
        rows_short = scale_x -  original_image_rows % scale_x
    #we will first get the number of cols short of a multiple of scale_y
    if original_image_cols % scale_y == 0:
        #we wont augument the image
        cols_short = 0
    else:
        cols_short = scale_y -  original_image_cols % scale_y
    #we will now pad the image with reflection padding
    #now if row_short is even, we will pad rows_short/2 on top and rows_short/2 on bottom
    #else we will pad rows_short/2 on top and rows_short/2 + 1 on bottom
    #similarly for cols_short
    #we will first check if rows_short is even
    if rows_short % 2 == 0:
        #pad rows_short/2 on top and rows_short/2 on bottom
        rows_top = int(rows_short/2)
        rows_bottom = rows_top
    else:
        #pad rows_short/2 on top and rows_short/2 + 1 on bottom
        rows_top = int(rows_short/2)
        rows_bottom = int(rows_short/2) + 1
    #we will first check if cols_short is even
    if cols_short % 2 == 0:
        #pad cols_short/2 on top and cols_short/2 on bottom
        cols_left = int(cols_short/2)
        cols_right = cols_left
    else:
        #pad cols_short/2 on top and cols_short/2 + 1 on bottom
        cols_left = int(cols_short/2)
        cols_right = int(cols_short/2) + 1
    #we will now pad the image
    image = np.pad(image, ((rows_top, rows_bottom), (cols_left, cols_right)), 'reflect')
    #return the image
    #if we have changed the shape of image, we will print the old and new shape
    if rows_short != 0 or cols_short != 0:
        print("image shape changed from ", image.shape, " to ", (original_image_rows, original_image_cols))
    return image

#----------------------- INOUT IMAGE AUGUMENTATION ENDS -----------------------#
#-----------------------                                -----------------------#




#--------------------------------- PRINTS AND PLOTS ---------------------------#
#---------------------------------              -------------------------------#

#define a function print_max_min which prints max min of a numpy array or PIL image
def print_max_min(image, string=''):
    #check if image is numpy array 
    if isinstance(image, np.ndarray) or isinstance(image, PIL.Image.Image):
        print('max of '+ string+' : ', np.max(image))
        print('min of '+ string+' : ', np.min(image))
    #check if image is torch tensor
    elif isinstance(image, torch.Tensor):
        print('max of '+ string+' : ', image.max())
        print('min of '+ string+' : ', image.min())
    else:
        #error
        print('error: image is not numpy array, PIL image or torch tensor')

#write a function to print image statistices : psnr, ssim, mse between target and reference image

def image_quality_metrics(img1, img2, data_range, verbose = True):
        
    psnr = metrics.peak_signal_noise_ratio(img1, img2, data_range=data_range)
    ssim = metrics.structural_similarity(img1, img2, data_range=data_range)
    # mse = metrics.mean_squared_error(img1, img2)
    #we will calculate euclidean distance between the two images
    euclidean_distance = np.linalg.norm(img1 - img2)
    #calculate log of the euclidean distance
    log_euclidean_distance = np.log(euclidean_distance)

    #we will print the psnr, ssim, mse
    if verbose:
        print("PSNR: ", psnr)
        print("SSIM: ", ssim)
        print("log l2 norm: ", log_euclidean_distance)
    #create a dictionary to store the metrics
    metrics_dict = {'psnr': psnr, 'ssim': ssim, 'log_l2': log_euclidean_distance}
    
    return metrics_dict













###########################  INITIALIZATION  #################################
###########################                #################################

# 1. 2D median filtering
### initialization
# we will write a function to get the initial iterate for ISTA update from the data

## Algorithm for Initilization:
#### input: image, mask, patch_radius
#### Step 1: All teh pixels in the mask are set to Inf
#### Step 2: go to each pixel in the mask and find the patch around it
#### Step 3: then sort the pixel intensities in the patch in ascending order
#### Step 4: if there is atleast one not Inf pixel in the patch, then we only keep the pixel intensities that are not NaN
#### Step 5: then we take the median of the selected pixel intensities
#### Step 6: replace the pixel in the mask with the median value
#### Step 7: repeat the steps 2 to 6 for all the pixels in the mask

# ------------------------ 2D MEDIAN FILTERING ----------------------------#
# ------------------------                  -------------------------------#
#we will write a function to implement median filtering to remove NaNs
#we will do it by looping over each pixel and if the pixel is NaN, we will consider a patch around it
#and get teh median of the patch and replace the Inf with the median
#for boundary conditions we will consider the pixels outside the image to be NaNs
#we will implement the above algorithm in iterative fashion
def Inf_median_filter_2D(image, kernal_radius=5):
    #kernal radius should be an odd number, if not we will make it odd by decrementing it by 1,
    #if kernal radius is 0, assert error
    assert kernal_radius > 0, 'kernal radius should be greater than 0'
    if kernal_radius % 2 == 0:
        kernal_radius = kernal_radius - 1
    #convert data type to float64
    image = image.astype(np.float64)
    #we will pad the image on margin on all 4 sides with Inf
    padded_image = np.pad(image, kernal_radius, 'constant', constant_values=np.inf)

    #we will get the indices in the image inpainted_image where value is Inf
    indices_infinity = np.where(np.isinf(image))
    #we will change the pixel values in the padded image and replace the Inf with the median of the patch
    #the we will extract the middle part of the padded image and return it as the filtered image
    #so we would have to constanly conver tindices back and forth from padded image to original image
    #we will iterate over the indices where the value is Inf, i.e indices_infinity
    for i in range(len(indices_infinity[0])):
        #get the indices of the patch
        row_index = indices_infinity[0][i]
        col_index = indices_infinity[1][i]
        #get the index on padded image
        row_index_padded = row_index + kernal_radius
        col_index_padded = col_index + kernal_radius
        #get the patch
        patch = padded_image[row_index_padded - kernal_radius:row_index_padded + kernal_radius + 1, \
            col_index_padded - kernal_radius:col_index_padded + kernal_radius + 1]
        #get the median of the patch
        #sort the patch
        patch = np.sort(patch, axis=None)
        #get the indexes where array has Inf
        indices_inf = np.where(np.isinf(patch))
        #if there is no value other than INf, we will continue
        if len(indices_inf[0]) == len(patch):
            continue
        #if there is no INf, then the index of array_end will be the last index
        elif len(indices_inf[0]) == 0:
            array_end = len(patch) - 1
        else:
            #get the index of first Inf
            array_end = np.where(np.isinf(patch))[0][0] - 1 
            #now array_end has to be >= 0 as we have already checked for the case where there is all Inf
            #and thus, the index of arrsy_end be non-negative
        #ge the array shortened to the index of array_end
        patch = patch[0:array_end + 1]
        #get the median
        median = np.median(patch)
        #replace Inf at teh index in padded image with median
        padded_image[row_index_padded, col_index_padded] = median
    #get the middle part of the padded image
    filtered_image = padded_image[kernal_radius:-kernal_radius, kernal_radius:-kernal_radius]
    return filtered_image


#we write a function to inintialize the inpainting reconstruction algorithm
#we will use the median filtering algorithm to fill the masked pixels
#input: observed_image, inpainting_mask, kernal_radius
def initialize_median_filter(observed_image, application_matrix, kernal_radius= 1):
    #create a copy of the observed image to initial_image
    initial_image = observed_image.copy()
    #we get the indices from the inpainting mask where the value is 0
    indices_mask = np.where(application_matrix == 0)
    #at all the indices  from indices_mask we will replace the value in observed_image with Inf
    initial_image[indices_mask] = np.Inf
    #we will have a loop in which the kernel radius will be increased by 2
    while kernal_radius <= min(initial_image.shape[0], initial_image.shape[1])+2:
        #we will apply the median filter to the initial_image
        initial_image = Inf_median_filter_2D(initial_image, kernal_radius)
        #we will check if there are any Inf in the initial_image
        if np.isinf(initial_image).any():
            #if there are Inf, we will increase the kernal radius by 2
            kernal_radius = kernal_radius + 2
        else:
            #if there are no Inf, we will break the loop
            break
    #we will return the initial_image
    return initial_image

# ------------------------ 2D MEDIAN FILTERING ENDS  ----------------------------#
# ------------------------                           ----------------------------#

###########################  INITIALIZATION ENDS #################################
###########################                      #################################
