import cv2


def extract_rgb_values(image_path, x, y):
    # Load the image
    image = cv2.imread(image_path)

    # Check if image is loaded successfully
    if image is None:
        print(f"Error: Unable to load image '{image_path}'")
        return

    # Get RGB values of the pixel at position (x, y)
    (b, g, r) = image[y, x]

    # Display the RGB values
    print(f"RGB values at position ({x}, {y}):")
    print(f"  Red: {r}")
    print(f"  Green: {g}")
    print(f"  Blue: {b}")


if __name__ == "__main__":
    # Replace 'image.jpg' with your image file path
    image_path = '/home/Student/Downloads/jan-folwarczny-ZXBPMnNVtlE-unsplash.jpg'

    # Specify the pixel coordinates (x, y) to extract RGB values
    x = 100
    y = 150

    # Call function to extract and display RGB values
    extract_rgb_values(image_path, x, y)
