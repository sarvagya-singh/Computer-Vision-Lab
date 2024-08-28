import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
input_image_path = '/home/Student/220962036/CVLab/Week2/360_F_602610481_nnwEMDdvwH4EACfmL2l6SRu1w2cVmphK.jpg'
image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded correctly
if image is None:
    raise ValueError("Image not found or unable to load.")

# Perform histogram equalization
equalized_image = cv2.equalizeHist(image)

# Save the result
output_image_path = 'equalized_image1.jpg'
cv2.imwrite(output_image_path, equalized_image)

# Display the images using matplotlib
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Equalized Image')
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')

plt.show()
