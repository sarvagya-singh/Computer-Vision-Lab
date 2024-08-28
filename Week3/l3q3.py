#3. Write a program to compare box filter and gaussian filter image outputs.

import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_box_filter(image, kernel_size):

    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    return cv2.filter2D(image, -1, kernel)


def apply_gaussian_filter(image, kernel_size, sigma):
  
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def compare_filters(image_path, kernel_size=5, sigma=1.0):
 
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")

  
    box_filtered_image = apply_box_filter(image, kernel_size)
    gaussian_filtered_image = apply_gaussian_filter(image, kernel_size, sigma)

   
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Box Filtered Image')
    plt.imshow(box_filtered_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Gaussian Filtered Image')
    plt.imshow(gaussian_filtered_image, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    input_image_path = "masaan.jpg"  # Replace with your input image path
    compare_filters(input_image_path)
