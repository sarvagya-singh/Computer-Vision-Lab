#2. Write a program to detect lines using Hough transform.

import numpy as np
import cv2
import matplotlib.pyplot as plt

def sobel_edge_detection(image):

 
    sobel_x = np.array([[ -1, 0, 1 ],
                        [ -2, 0, 2 ],
                        [ -1, 0, 1 ]])

    sobel_y = np.array([[ -1, -2, -1 ],
                        [  0,  0,  0 ],
                        [  1,  2,  1 ]])

    grad_x = cv2.filter2D(image, cv2.CV_64F, sobel_x)
    grad_y = cv2.filter2D(image, cv2.CV_64F, sobel_y)


    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = np.uint8(np.clip(magnitude, 0, 255))

    return magnitude

def hough_transform(image):

    
    diag_len = int(np.sqrt(image.shape[0]**2 + image.shape[1]**2))
    rho_res = 1
    theta_res = np.deg2rad(1)

 
    thetas = np.arange(-90, 90, 1)
    rhos = np.arange(-diag_len, diag_len, rho_res)
    hough_space = np.zeros((len(rhos), len(thetas)), dtype=np.int)

   
    y_indices, x_indices = np.nonzero(image)

    for i in range(len(x_indices)):
        x = x_indices[i]
        y = y_indices[i]
        for theta_index in range(len(thetas)):
            theta = np.deg2rad(thetas[theta_index])
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            rho_index = np.argmin(np.abs(rhos - rho))
            hough_space[rho_index, theta_index] += 1

    return hough_space, thetas, rhos

def detect_lines(hough_space, thetas, rhos, threshold):

    lines = []
    for i in range(hough_space.shape[0]):
        for j in range(hough_space.shape[1]):
            if hough_space[i, j] > threshold:
                rho = rhos[i]
                theta = thetas[j]
                lines.append((rho, theta))
    return lines

def draw_lines(image, lines):
 
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height, width = image.shape
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return result_image


input_image_path = 'masaan.jpg'
output_image_path = 'detected_lines.jpg'
threshold = 100


image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"The image at path {input_image_path} was not found.")


edges = sobel_edge_detection(image)


hough_space, thetas, rhos = hough_transform(edges)


lines = detect_lines(hough_space, thetas, rhos, threshold)


result_image = draw_lines(image, lines)

cv2.imwrite(output_image_path, result_image)
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title('Detected Lines')
plt.show()

