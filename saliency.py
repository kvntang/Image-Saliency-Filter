import cv2
import numpy as np

# Load the input image
image = cv2.imread('p2.jpg')  # Replace 'image.jpg' with your image filename

# Check if the image was successfully loaded
if image is None:
    print("Could not read the image.")
    exit()

# Choose one of the saliency detectors:
# Option 1: Static Saliency Spectral Residual
# saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

# Option 2: Static Saliency Fine-Grained
saliency = cv2.saliency.StaticSaliencyFineGrained_create()

# Compute the saliency map
(success, saliencyMap) = saliency.computeSaliency(image)

if not success:
    print("Saliency computation failed.")
    exit()

# Normalize the saliency map to the range [0, 255]
saliencyMap = (saliencyMap * 255).astype("uint8")

# Threshold the saliency map to create a binary map
_, threshMap = cv2.threshold(saliencyMap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Optionally invert the threshold map if needed
# threshMap = cv2.bitwise_not(threshMap)

# Save the binary saliency map
cv2.imwrite('saliency_binary.png', threshMap)

# Apply the binary map as a mask to the original image to get the cleaned-up version
masked_image = cv2.bitwise_and(image, image, mask=threshMap)

# Save the cleaned-up image
cv2.imwrite('cleaned_image.png', masked_image)

# Display the images (optional)
cv2.imshow("Original Image", image)
cv2.imshow("Saliency Map", saliencyMap)
cv2.imshow("Thresholded Saliency Map", threshMap)
cv2.imshow("Cleaned-up Image", masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
