#1. Write a program to read an image and perform unsharp masking.

import cv2
import numpy as np


def create_gaussian_kernel(size, sigma):
  
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) *
                     np.exp(-(((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2))),
        (size, size)
    )
    return kernel / np.sum(kernel)


def unsharp_masking(image_path, kernel_size=5, sigma=1.0, strength=1.5, display_width=600):

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

 
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)

    # Apply Gaussian blur using the custom kernel
    blurred_image = cv2.filter2D(gray_image, -1, gaussian_kernel)

    # Create the unsharp mask
    unsharp_mask = cv2.addWeighted(gray_image, 1.0 + strength, blurred_image, -strength, 0)

    # Convert the grayscale result back to BGR (3 channels)
    unsharp_mask_bgr = cv2.cvtColor(unsharp_mask, cv2.COLOR_GRAY2BGR)

    # Resize images for display
    def resize_image(image, width):
        height, original_width = image.shape[:2]
        aspect_ratio = height / original_width
        new_height = int(width * aspect_ratio)
        resized_image = cv2.resize(image, (width, new_height))
        return resized_image

    image_resized = resize_image(image, display_width)
    unsharp_mask_bgr_resized = resize_image(unsharp_mask_bgr, display_width)


    combined_image = np.hstack((image_resized, unsharp_mask_bgr_resized))

  
    cv2.imshow('Original and Sharpened Images', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    input_image_path = "masaan.jpg"
    unsharp_masking(input_image_path)
