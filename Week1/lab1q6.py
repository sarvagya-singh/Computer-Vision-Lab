import cv2


def rotate_image(image_path, angle):
    # Load the image
    image = cv2.imread(image_path)

    # Check if image is loaded successfully
    if image is None:
        print(f"Error: Unable to load image '{image_path}'")
        return

    # Get image dimensions
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Rotate the image by specified angle
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))

    # Display the original and rotated images (optional)
    cv2.imshow("Original Image", image)
    cv2.imshow("Rotated Image", rotated)
    cv2.waitKey(0)  # Wait for any key press before closing the window
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Replace 'image.jpg' with your image file path
    image_path = '/home/Student/Downloads/jan-folwarczny-ZXBPMnNVtlE-unsplash.jpg'

    # Specify the rotation angle in degrees
    angle = 45

    # Call function to rotate the image
    rotate_image(image_path, angle)
