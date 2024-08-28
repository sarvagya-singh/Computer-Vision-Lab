import cv2


def draw_rectangle(image_path, pt1, pt2):
    # Load the image
    image = cv2.imread(image_path)

    # Check if image is loaded successfully
    if image is None:
        print(f"Error: Unable to load image '{image_path}'")
        return

    # Draw a rectangle on the image
    color = (0, 255, 0)  # BGR format (here, green color)
    thickness = 2  # Thickness of the rectangle border (optional)
    cv2.rectangle(image, pt1, pt2, color, thickness)

    # Display the image with the rectangle
    cv2.imshow('Image with Rectangle', image)
    cv2.waitKey(0)  # Wait for any key press before closing the window
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Replace 'image.jpg' with your image file path
    image_path = '/home/Student/Downloads/jan-folwarczny-ZXBPMnNVtlE-unsplash.jpg'

    # Specify the top-left and bottom-right corners of the rectangle
    pt1 = (50, 50)  # (x1, y1)
    pt2 = (200, 150)  # (x2, y2)

    # Call function to draw rectangle on the image
    draw_rectangle(image_path, pt1, pt2)
