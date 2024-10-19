import cv2
import numpy as np

# Read image from root directory
image = cv2.imread('p2.jpg')  # Replace 'image.jpg' with your image filename

# Check if image was successfully loaded
if image is None:
    print("Could not read the image.")
    exit()

# Initialize the static saliency spectral residual detector
saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

# Compute the saliency map
(success, saliencyMap) = saliency.computeSaliency(image)

# Convert the saliency map to 8-bit grayscale image
saliencyMap = (saliencyMap * 255).astype("uint8")

# Threshold the saliency map to get the binary mask
_, threshMap = cv2.threshold(saliencyMap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Save the binary mask image
cv2.imwrite('saliency_threshold22222.png', threshMap)

# Apply the mask to the original image to get the cleaned-up version
masked_image = cv2.bitwise_and(image, image, mask=threshMap)

# Save the cleaned-up image
cv2.imwrite('cleaned_image2222222.png', masked_image)
