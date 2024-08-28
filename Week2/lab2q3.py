import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_histogram(image):
    """Compute histogram of an image."""
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    return hist


def compute_cdf(hist):
    """Compute cumulative distribution function (CDF) from histogram."""
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]  # Normalize to [0, 1]
    return cdf_normalized


def histogram_equalization(image):
    """Perform histogram equalization on the input image."""
    # Compute histogram and CDF
    hist = compute_histogram(image)
    cdf = compute_cdf(hist)

    # Create a mapping of pixel values
    # Scale CDF to 0-255
    cdf_scaled = (cdf * 255).astype('uint8')

    # Apply the mapping to the image
    equalized_image = cdf_scaled[image]
    return equalized_image


# Load the image
input_image_path = '4c37d6d0-f3dc-11e9-87ad-fce8e65242a6_image_hires_190248.jpg'  # Replace with your image path
image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded correctly
if image is None:
    raise ValueError("Image not found or unable to load.")

# Perform histogram equalization
equalized_image = histogram_equalization(image)

# Save the result
output_image_path = 'equalized_image2.jpg'
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

'''
import cv2
import matplotlib.pyplot as plt

def resize_image(image, width, height):
    """Resize the image to the specified width and height."""
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized_image

def crop_image(image, x, y, width, height):
    """Crop the image to the specified region."""
    cropped_image = image[y:y+height, x:x+width]
    return cropped_image

# Load the image
input_image_path = '4c37d6d0-f3dc-11e9-87ad-fce8e65242a6_image_hires_190248.jpg'  # Replace with your image path
image = cv2.imread(input_image_path)

# Check if the image was loaded correctly
if image is None:
    raise ValueError("Image not found or unable to load.")

# Resize parameters
resize_width = 200
resize_height = 200

# Crop parameters (x, y, width, height)
crop_x = 50
crop_y = 50
crop_width = 100
crop_height = 100

# Perform resizing
resized_image = resize_image(image, resize_width, resize_height)

# Perform cropping
cropped_image = crop_image(image, crop_x, crop_y, crop_width, crop_height)

# Save the results
cv2.imwrite('resized_image.jpg', resized_image)
cv2.imwrite('cropped_image.jpg', cropped_image)

# Display the images using matplotlib
plt.figure(figsize=(15, 10))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Resized Image')
plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Cropped Image')
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
'''