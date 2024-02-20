import cv2
import numpy as np

# Load an image
image = cv2.imread('data acquisition/images/robot_0.png')

# Convert the image to grayscale as edge detection requires single channel image formats
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

### Edge Detection ###
edges = cv2.Canny(gray, 100, 200)  # Adjust these thresholds based on your image

### Segmentation ###
# Simple thresholding (assumes a somewhat uniform background)
# You might need to adjust the threshold value based on your images
_, segmented = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

### Keypoint Detection ###
# Shi-Tomasi Corner Detector & Good Features to Track
corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
corners = np.int0(corners)

# Draw detected corners on the original image
for i in corners:
    x, y = i.ravel()
    cv2.circle(image, (x, y), 3, (255, 0, 0), -1)  # Draw red circles on detected corners

### Display Results ###
cv2.imshow('Edges', edges)
cv2.imshow('Segmented', segmented)
cv2.imshow('Keypoints', image)  # Display keypoints on the original image

# Wait for a key press and then terminate the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
