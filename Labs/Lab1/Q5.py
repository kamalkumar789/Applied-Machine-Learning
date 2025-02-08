import cv2
import numpy as np

# Load the image in BGR format
image = cv2.imread("/home/kamal/Applied Machine Learning/Labs/Lab1/istockphoto-930082108-612x612.jpg")


# grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert the image to grayscale manually using weighted sum
grey_img = 0.30 * image[:, :, 2] + 0.59 * image[:, :, 1] + 0.11 * image[:, :, 0]  # Correct BGR order

# Convert to uint8 (Required for OpenCV display)
grey_img = np.uint8(grey_img)

# Show the grayscale image
cv2.imshow("Grayscale Image", grey_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
