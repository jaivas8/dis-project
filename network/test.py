import cv2
import numpy as np

# Load the image
image = cv2.imread('network/data/blended_image/robot_0.png')

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Threshold the image to create a binary image
_, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('Binary Image', binary)
cv2.waitKey(0)

# Find contours from the binary image
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assuming the largest contour is the object of interest
largest_contour = max(contours, key=cv2.contourArea)

# Create a mask of the largest contour
mask = np.zeros_like(gray)
cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
cv2.imshow('Mask', mask)
cv2.waitKey(0)

# Use the mask to extract the object
result = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Result",result)
# Save or display the result
cv2.waitKey(0)
cv2.imwrite('path_to_your_output_image.png', result)