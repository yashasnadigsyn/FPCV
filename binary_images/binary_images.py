import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

img = cv2.imread('chika.jpeg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

if len(img.shape) > 2:
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    gray_image = img

_, binary_image_raw = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image_raw, 8, cv2.CV_32S)

if num_labels > 1:
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_component_index = np.argmax(areas)
    main_object_label = largest_component_index + 1
else:
    binary_image = np.zeros_like(binary_image_raw, dtype=np.uint8)

output_mask = np.zeros_like(binary_image_raw, dtype=np.uint8)
output_mask[labels == main_object_label] = 1
binary_image = output_mask


## Area of the image
area = np.sum(binary_image)
print(f"Area: {area} pixels")

## Centre of Area
rows, cols = binary_image.shape
i_coords, j_coords = np.indices((rows, cols))

if area == 0:
    print("No object found to analyze.")
    exit()

j_bar = np.sum(binary_image * j_coords) / area
i_bar = np.sum(binary_image * i_coords) / area
print(f"Centroid (x, y): ({j_bar:.2f}, {i_bar:.2f})")

## Orientation and Roundedness

### Second Moments around the origin
a_dash = np.sum(binary_image * np.square(j_coords))
c_dash = np.sum(binary_image * np.square(i_coords))
b_dash = 2 * np.sum(binary_image * i_coords * j_coords)

### Translate to Central Moments
a = a_dash - area * (j_bar ** 2)
c = c_dash - area * (i_bar ** 2)
b = b_dash - 2 * area * j_bar * i_bar

### Calculate Orientation
two_theta = math.atan2(b, a - c)
theta1 = two_theta / 2.0
print(f"Orientation: {math.degrees(theta1):.2f} degrees")

### Calculate theta2 from theta1
theta2 = theta1 + math.pi / 2

### Calculate e1 and e2
e1 = a * math.sin(theta1)**2 - b * math.sin(theta1) * math.cos(theta1) + c * math.cos(theta1)**2
e2 = a * math.sin(theta2)**2 - b * math.sin(theta2) * math.cos(theta2) + c * math.cos(theta2)**2

### Find E_min and E_max
e_min = min(e1, e2)
e_max = max(e1, e2)

### Calculate Roundedness
if e_max == 0:
    roundedness = 1.0
else:
    roundedness = e_min / e_max
print(f"Roundedness: {roundedness:.2f}")


fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1, 0.8]})

## Original Images
axes[0].imshow(img_rgb)
axes[0].set_title('Original Image')
axes[0].set_xticks([])
axes[0].set_yticks([])

## Edited Image
axes[1].imshow(binary_image, cmap='gray')
axes[1].set_title('Orientation and Centroid')
axes[1].set_xticks([])
axes[1].set_yticks([])

if area > 0:
    axes[1].plot(j_bar, i_bar, 'r.', markersize=15)

    length = max(rows, cols) * 0.8
    cos_theta = math.cos(theta1)
    sin_theta = math.sin(theta1)
    p1_x = j_bar + length * cos_theta
    p1_y = i_bar + length * sin_theta
    p2_x = j_bar - length * cos_theta
    p2_y = i_bar - length * sin_theta
    axes[1].plot([p1_x, p2_x], [p1_y, p2_y], 'r-', linewidth=2)

    axes[1].set_xlim(0, cols)
    axes[1].set_ylim(rows, 0)

axes[2].axis('off')

info_text = (
    f"Object Properties: \n"
    f"Area: {area} pixels\n\n"
    f"Centroid (x, y):\n({j_bar:.2f}, {i_bar:.2f})\n\n"
    f"Orientation:\n{math.degrees(theta1):.2f}°\n\n"
    f"Roundedness:\n{roundedness:.2f}"
)

axes[2].text(
    x=0.05,            
    y=0.75,           
    s=info_text,       
    ha='left',         
    va='top',           
    fontsize=12,
    wrap=True          
)

plt.tight_layout()
plt.show()