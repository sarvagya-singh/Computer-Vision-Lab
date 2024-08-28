# Lab 4: Implementation of Image Segmentation Methods
#1. Write a program to create binary images using thresholding methods.

import numpy as np
import cv2

def binary_threshold(image, threshold_value):

    binary_image = np.where(image >= threshold_value, 255, 0)
    return binary_image

def binary_threshold_inverted(image, threshold_value):

    binary_image = np.where(image < threshold_value, 255, 0)
    return binary_image

def truncated_threshold(image, threshold_value):

    truncated_image = np.clip(image, 0, threshold_value)
    return truncated_image

def set_to_zero(image, threshold_value):

    modified_image = np.where(image < threshold_value, 0, image)
    return modified_image

def set_to_zero_inverted(image, threshold_value):

    modified_image = np.where(image >= threshold_value, 0, image)
    return modified_image

def process_image(image_path, threshold_value, output_path_prefix):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"The image at path {image_path} was not found.")

 
    binary_img = binary_threshold(image, threshold_value)
    binary_inv_img = binary_threshold_inverted(image, threshold_value)
    truncated_img = truncated_threshold(image, threshold_value)
    set_zero_img = set_to_zero(image, threshold_value)
    set_zero_inv_img = set_to_zero_inverted(image, threshold_value)

 

    cv2.imwrite(f'{output_path_prefix}_binary.jpg', binary_img)
    cv2.imwrite(f'{output_path_prefix}_binary_inv.jpg', binary_inv_img)
    cv2.imwrite(f'{output_path_prefix}_truncated.jpg', truncated_img)
    cv2.imwrite(f'{output_path_prefix}_set_to_zero.jpg', set_zero_img)
    cv2.imwrite(f'{output_path_prefix}_set_to_zero_inv.jpg', set_zero_inv_img)

    print("Processing completed. Check the output images.")


input_image_path = 'masaan.jpg'
threshold_value = 128
output_image_prefix = 'output_image'

process_image(input_image_path, threshold_value, output_image_prefix)
