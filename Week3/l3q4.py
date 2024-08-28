#4. Write a program to detect edges in a image.

import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_edges(image_path, low_threshold=100, high_threshold=200):
    """
    Detect edges in an image using the Canny edge detection algorithm.

    Parameters:
        image_path (str): Path to the input image.
        low_threshold (int): Low threshold for the Canny edge detection.
        high_threshold (int): High threshold for the Canny edge detection.
    """
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")

    # Apply Canny edge detection
    edges = cv2.Canny(image, low_threshold, high_threshold)

    # Display the results
    plt.figure(figsize=(8, 8))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Edges Detected')
    plt.imshow(edges, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    input_image_path = "/home/Student/220962036/CVLab/Week2/360_F_602610481_nnwEMDdvwH4EACfmL2l6SRu1w2cVmphK.jpg"  # Replace with your input image path
    detect_edges(input_image_path)
