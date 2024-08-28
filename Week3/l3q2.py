#2. Write a program to obtain gradient of an image.


import cv2
import numpy as np
import matplotlib.pyplot as plt


def gradient_kernels():
   
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]], dtype=np.float32)

    kernel_y = np.array([[-1, -1, -1],
                         [0, 0, 0],
                         [1, 1, 1]], dtype=np.float32)

    return kernel_x, kernel_y


def apply_kernel(image, kernel):
    
    return cv2.filter2D(image, -1, kernel)


def compute_gradients(image_path):
    

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")


    kernel_x, kernel_y = gradient_kernels()

 
    grad_x = apply_kernel(image, kernel_x)
    grad_y = apply_kernel(image, kernel_y)

    # Compute gradient magnitude
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_magnitude = cv2.normalize(grad_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Compute gradient direction
    grad_direction = np.arctan2(grad_y, grad_x) * (180 / np.pi)
    grad_direction = (grad_direction + 180) % 180  # Normalize to [0, 180]
    grad_direction = grad_direction.astype(np.uint8)

    
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Gradient Magnitude')
    plt.imshow(grad_magnitude, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Gradient Direction')
    plt.imshow(grad_direction, cmap='hsv')
    plt.axis('off')

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    input_image_path = "masaan.jpg"
    compute_gradients(input_image_path)

