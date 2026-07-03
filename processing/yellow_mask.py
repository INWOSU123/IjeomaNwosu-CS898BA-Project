import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

folder = "dataset/bus"

# Get the first readable bus image
img = None

for file in os.listdir(folder):
    path = os.path.join(folder, file)
    temp = cv2.imread(path)

    if temp is not None:
        img = temp
        print("Using image:", file)
        break

if img is None:
    print("No readable image found.")
    exit()

# Resize
img = cv2.resize(img, (128, 128))

# Convert BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# School bus yellow range
lower_yellow = np.array([15, 80, 80])
upper_yellow = np.array([40, 255, 255])

# Create yellow mask
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Morphological cleanup
kernel = np.ones((5, 5), np.uint8)

opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

# Show results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("HSV Yellow Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(closed, cmap="gray")
plt.title("Morphology Cleaned")
plt.axis("off")

plt.show()