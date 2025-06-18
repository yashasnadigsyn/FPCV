import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

image_path = "kitty.png"


img_bgr = cv2.imread(image_path)

if len(img_bgr.shape) > 2:
    gray_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
else:
    gray_image = img_bgr

fig_hist, axes_hist = plt.subplots(1, 2, figsize=(12, 5))
fig_hist.suptitle('Histogram Analysis', fontsize=16)

axes_hist[0].imshow(gray_image, cmap='gray')
axes_hist[0].set_title('Grayscale Image')
axes_hist[0].axis('off')

axes_hist[1].hist(gray_image.flatten(), bins=256, range=[0, 256], color='gray', log=True)
axes_hist[1].set_title('Logarithmic Intensity Histogram')
axes_hist[1].set_xlabel('Pixel Intensity (0=Black, 255=White)')
axes_hist[1].set_ylabel('Number of Pixels (Log Scale)')
axes_hist[1].set_xlim([0, 256])

otsu_threshold, _ = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu's method calculated optimal threshold = {int(otsu_threshold)}")

axes_hist[1].axvline(otsu_threshold, color='g', linestyle='dashed', linewidth=2, label=f"Otsu's Threshold = {int(otsu_threshold)}")
axes_hist[1].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()