from PIL import Image
import os

path = os.path.join(os.getcwd(), 'images')
images = os.listdir(path)
print("converting...")
for img_name in images:
    try:
        img_path = path + "\\" + img_name
        img = Image.open(img_path)
        png_img_name = path + "_png\\" + img_name.replace(".ppm", ".png")
        img.save(png_img_name)
    except:
        print("Error loading image")

print("Converting Finished.")
quit()
