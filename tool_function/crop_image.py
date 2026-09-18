import os
import cv2
import zipfile
import numpy as np

# 1. 设置输入和输出文件夹路径
input_folder = r"E:\pythondemo\number_dectect_project\imagedata\raw_image\image3.jpg"
output_folder = r"E:\pythondemo\number_dectect_project\imagedata\resized_32x32"
img = cv2.imread(input_folder )

if img is None:
    # Try finding image file in current directory
    for f in os.listdir("."):
        if f.endswith(".jpg") or f.endswith(".png"):
            img_path = f
            img = cv2.imread(img_path)
            break

h, w, _ = img.shape
print(f"Image dimensions: {w}x{h}")

# The image contains a 10x10 grid of numbers separated by grey grid lines.
# Let's inspect the grid dimensions/positions or crop based on 10x10 grid.
rows = 10
cols = 10

# 2. 如果输出文件夹不存在，则自动新建
os.makedirs(output_folder, exist_ok=True)

cell_h = h / rows
cell_w = w / cols

saved_files = []

for r in range(rows):
    for c in range(cols):
        # Calculate bounding coordinates for each cell
        y1 = int(r * cell_h)
        y2 = int((r + 1) * cell_h)
        x1 = int(c * cell_w)
        x2 = int((c + 1) * cell_w)

        # Crop cell
        cell = img[y1:y2, x1:x2]

        # Resize cell exactly to 32x32 as requested by user
        cell_32x32 = cv2.resize(cell, (32, 32), interpolation=cv2.INTER_AREA)

        filename = os.path.join(output_folder, f"digit_row{r + 1}_col{c + 1}.png")
        cv2.imwrite(filename, cell_32x32)
        saved_files.append(filename)

# Package all 100 images into a single zip file for convenient download
zip_filename = "digits_32x32_all.zip"
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for file in saved_files:
        zipf.write(file, os.path.basename(file))

print(f"Successfully processed {len(saved_files)} images and zipped into {zip_filename}.")