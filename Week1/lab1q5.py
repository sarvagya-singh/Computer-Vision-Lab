import cv2


def resize_image(image_path, width=None, height=None):
    # Load the image
    image = cv2.imread(image_path)

    # Check if image is loaded successfully
    if image is None:
        print(f"Error: Unable to load image '{image_path}'")
        return

    # Define new dimensions
    if width is None and height is None:
        print("Error: Please specify either width or height for resizing.")
        return

    if width is None:
        # Calculate proportional height if only width is specified
        ratio = height / image.shape[0]
        dim = (int(image.shape[1] * ratio), height)
    elif height is None:
        # Calculate proportional width if only height is specified
        ratio = width / image.shape[1]
        dim = (width, int(image.shape[0] * ratio))
    else:
        # Resize to specified width and height
        dim = (width, height)

    # Resize the image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Display the original and resized images (optional)
    cv2.imshow("Original Image", image)
    cv2.imshow("Resized Image", resized)
    cv2.waitKey(0)  # Wait for any key press before closing the window
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Replace 'image.jpg' with your image file path
    image_path = '/home/Student/Downloads/jan-folwarczny-ZXBPMnNVtlE-unsplash.jpg'

    # Specify new width and/or height for resizing (in pixels)
    width = 400
    height = 300

    # Call function to resize the image
    resize_image(image_path, width, height)
