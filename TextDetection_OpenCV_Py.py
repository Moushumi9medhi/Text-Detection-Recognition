"""
Text Region Detection 

Dependencies:
- OpenCV (cv2)
"""

import cv2

# --- Configuration ---
INPUT_IMAGE = "Your_Image.jpg"
OUTPUT_IMAGE = "text_regions_detected.jpg"

# --- Load and preprocess image ---
img_original = cv2.imread(INPUT_IMAGE)
img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

# Apply binary inverse thresholding
_, img_binary_inv = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY_INV)

 # Create a cross-shaped kernel to help connect characters
kernel_shape = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
img_dilated = cv2.dilate(img_binary_inv, kernel_shape, iterations=4)

# --- Find contours ---
# Retrieves only external contours, keeping hierarchy flat
contours, _ = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # Skip overly large areas (likely non-text)
    if h > 50 and w > 300:
        continue

    # Skip very small regions (likely noise)
    if h < 4 or w < 4:
        continue

    
    cv2.rectangle(img_original, (x, y), (x + w, y + h), (255, 0, 255), 2)

# --- Save result ---
cv2.imwrite(OUTPUT_IMAGE, img_original)
print(f"Text region detection complete. Output saved as '{OUTPUT_IMAGE}'")
