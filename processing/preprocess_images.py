import cv2
import os
import matplotlib.pyplot as plt


folder="dataset/bus"

file=os.listdir(folder)[0]

img=cv2.imread(
    os.path.join(
        folder,
        file
    )
)

img=cv2.resize(
    img,
    (128,128)
)


hsv=cv2.cvtColor(
    img,
    cv2.COLOR_BGR2HSV
)


clahe=cv2.createCLAHE(
    clipLimit=2
)


hsv[:,:,2]=clahe.apply(
    hsv[:,:,2]
)


result=cv2.cvtColor(
    hsv,
    cv2.COLOR_HSV2RGB
)


plt.figure(
figsize=(10,5)
)

plt.subplot(1,2,1)

plt.imshow(
cv2.cvtColor(
img,
cv2.COLOR_BGR2RGB
)
)

plt.title(
"Original"
)

plt.axis("off")


plt.subplot(1,2,2)

plt.imshow(result)

plt.title(
"CLAHE"
)

plt.axis("off")

plt.show()