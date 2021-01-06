import numpy as np
from PIL import Image

input_image_path="./testing_images.png"
color_image = Image.open(input_image_path)
bw = color_image.convert('L')
bw.save("./testing_imagesBW.png")
