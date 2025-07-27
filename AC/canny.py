import cv2
import os
# python canny.py   
input_folder = "datasets/ZZCX_3_3/test/RGB_HQ"
output_folder = "datasets/ZZCX_3_3/test/edge"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


for filename in os.listdir(input_folder):
    if filename.endswith(('.png')):  
        input_path = os.path.join(input_folder, filename)
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            edges = cv2.Canny(img, 32, 64)                                # HR是  128 200
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, edges)
        else:
            print(f"无法正确读取图片 {input_path}")