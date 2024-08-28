#5.Implement Canny edge detection algorithm.

import cv2
import numpy as np
import matplotlib.pyplot as plt


def gaussian_blur(image, kernel_size=5, sigma=1.0):

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def compute_gradients(image):

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float32)

    grad_x = cv2.filter2D(image, cv2.CV_64F, sobel_x)
    grad_y = cv2.filter2D(image, cv2.CV_64F, sobel_y)

    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_direction = np.arctan2(grad_y, grad_x) * (180 / np.pi)
    grad_direction = (grad_direction + 180) % 180  # Normalize to [0, 180]

    return grad_x, grad_y, grad_magnitude, grad_direction


def non_max_suppression(grad_magnitude, grad_direction):

    height, width = grad_magnitude.shape
    suppressed = np.zeros_like(grad_magnitude)

    angle = grad_direction / 180.0 * np.pi  # Convert to radians

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            q = 255
            r = 255

            # Horizontal edge
            if (0 <= angle[i, j] < np.pi / 8) or (15 * np.pi / 8 <= angle[i, j] < 2 * np.pi):
                q = grad_magnitude[i, j + 1]
                r = grad_magnitude[i, j - 1]
            # Diagonal edge
            elif np.pi / 8 <= angle[i, j] < 3 * np.pi / 8:
                q = grad_magnitude[i + 1, j + 1]
                r = grad_magnitude[i - 1, j - 1]
            # Vertical edge
            elif 3 * np.pi / 8 <= angle[i, j] < 5 * np.pi / 8:
                q = grad_magnitude[i + 1, j]
                r = grad_magnitude[i - 1, j]
            # Other diagonal edge
            elif 5 * np.pi / 8 <= angle[i, j] < 7 * np.pi / 8:
                q = grad_magnitude[i - 1, j + 1]
                r = grad_magnitude[i + 1, j - 1]

            if grad_magnitude[i, j] >= q and grad_magnitude[i, j] >= r:
                suppressed[i, j] = grad_magnitude[i, j]

    return suppressed


def double_threshold(image, low_threshold, high_threshold):
 
    strong = 255
    weak = 75

    edges = np.zeros_like(image, dtype=np.uint8)
    strong_edges = (image >= high_threshold)
    weak_edges = ((image >= low_threshold) & (image < high_threshold))

    edges[strong_edges] = strong
    edges[weak_edges] = weak

    return edges


def edge_tracking_by_hysteresis(edges):

    strong = 255
    weak = 75
    height, width = edges.shape

    final_edges = np.copy(edges)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if edges[i, j] == weak:
                if ((edges[i + 1, j] == strong) or (edges[i - 1, j] == strong) or
                        (edges[i, j + 1] == strong) or (edges[i, j - 1] == strong) or
                        (edges[i + 1, j + 1] == strong) or (edges[i - 1, j - 1] == strong) or
                        (edges[i + 1, j - 1] == strong) or (edges[i - 1, j + 1] == strong)):
                    final_edges[i, j] = strong
                else:
                    final_edges[i, j] = 0

    return final_edges


def canny_edge_detection(image_path, kernel_size=5, sigma=1.0, low_threshold=50, high_threshold=150):

    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")

    # Step 1: Noise Reduction
    blurred_image = gaussian_blur(image, kernel_size, sigma)

    # Step 2: Compute Gradients
    grad_x, grad_y, grad_magnitude, grad_direction = compute_gradients(blurred_image)

    # Step 3: Non-Maximum Suppression
    non_max_suppressed_image = non_max_suppression(grad_magnitude, grad_direction)

    # Step 4: Double Threshold
    edges = double_threshold(non_max_suppressed_image, low_threshold, high_threshold)


    final_edges = edge_tracking_by_hysteresis(edges)

 
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Canny Edges')
    plt.imshow(final_edges, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    input_image_path = "/home/Student/220962036/CVLab/Week2/360_F_602610481_nnwEMDdvwH4EACfmL2l6SRu1w2cVmphK.jpg"
    canny_edge_detection(input_image_path)
