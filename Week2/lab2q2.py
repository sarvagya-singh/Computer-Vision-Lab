import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_histogram(image):
    """Compute histogram of an image."""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist


def compute_cdf(hist):
    """Compute cumulative distribution function (CDF) from histogram."""
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]  # Normalize to [0, 1]
    return cdf_normalized


def histogram_matching(source, reference):
    """Perform histogram matching to match the source image to the reference histogram."""
    # Compute histograms and CDFs
    src_hist = compute_histogram(source)
    ref_hist = compute_histogram(reference)

    src_cdf = compute_cdf(src_hist)
    ref_cdf = compute_cdf(ref_hist)

    # Create a mapping of pixel values from source to reference
    mapping = np.zeros(256)
    src_idx = 0
    for ref_idx in range(256):
        while src_cdf[src_idx] < ref_cdf[ref_idx] and src_idx < 255:
            src_idx += 1
        mapping[ref_idx] = src_idx

    # Apply the mapping to the source image
    matched_image = mapping[source.astype('uint8')]
    return matched_image.astype('uint8')


# Load the images
input_image_path = '/home/Student/220962036/CVLab/Week2/equalized_image.jpg'  # Replace with your input image path
reference_image_path = '/home/Student/220962036/CVLab/Week2/4c37d6d0-f3dc-11e9-87ad-fce8e65242a6_image_hires_190248.jpg'  # Replace with your reference image path

input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

# Check if the images were loaded correctly
if input_image is None or reference_image is None:
    raise ValueError("One or both images not found or unable to load.")

# Perform histogram specification
matched_image = histogram_matching(input_image, reference_image)

# Save the result
output_image_path = 'matched_image.jpg'
cv2.imwrite(output_image_path, matched_image)

# Display the images using matplotlib
plt.figure(figsize=(15, 10))

plt.subplot(1, 3, 1)
plt.title('Input Image')
plt.imshow(input_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Reference Image')
plt.imshow(reference_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Matched Image')
plt.imshow(matched_image, cmap='gray')
plt.axis('off')

plt.show()
