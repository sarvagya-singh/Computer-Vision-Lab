#3. Write a program to segment an image based on colour.

import numpy as np
import cv2
import matplotlib.pyplot as plt

def rgb_to_hsv(image):
  
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def create_color_mask(hsv_image, lower_bound, upper_bound):
   
    lower_bound = np.array(lower_bound, dtype=np.uint8)
    upper_bound = np.array(upper_bound, dtype=np.uint8)

    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    return mask

def apply_mask(image, mask):
    
    return cv2.bitwise_and(image, image, mask=mask)

def plot_results(original_image, mask, segmented_image):
   
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 3, 2)
    plt.title('Mask')
    plt.imshow(mask, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Segmented Image')
    plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))

    plt.show()


input_image_path = 'masaan.jpg'
lower_color_bound = (30, 100, 100) 
upper_color_bound = (90, 255, 255)

image = cv2.imread(input_image_path)
if image is None:
    raise FileNotFoundError(f"The image at path {input_image_path} was not found.")


hsv_image = rgb_to_hsv(image)


mask = create_color_mask(hsv_image, lower_color_bound, upper_color_bound)


segmented_image = apply_mask(image, mask)


plot_results(image, mask, segmented_image)
