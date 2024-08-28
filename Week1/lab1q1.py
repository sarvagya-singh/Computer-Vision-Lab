import cv2

img_grayscale = cv2.imread('/home/Student/Downloads/jan-folwarczny-ZXBPMnNVtlE-unsplash.jpg')

cv2.imshow('grayscale image', img_grayscale)

cv2.waitKey(0)

cv2.destroyAllWindows()

cv2.imwrite('d:\grayscale1.jpg',img_grayscale)