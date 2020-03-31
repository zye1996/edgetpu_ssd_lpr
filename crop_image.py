import tensorflow as tf
import cv2
import os
import io
import numpy as np
from PIL import Image, ImageDraw


img_name = "00205459770115-90_85-352&516_448&547-438&539_362&540_360&519_436&518-0_0_22_10_26_29_24-128-7.jpg"
path = "images_rec/representative_images"

with tf.io.gfile.GFile(os.path.join(path, '{}'.format(img_name)), 'rb') as fid:
    encoded_jpg = fid.read()
encoded_jpg_io = io.BytesIO(encoded_jpg)
image = Image.open(encoded_jpg_io)

print(len(encoded_jpg))

buf = io.BytesIO()
image.save(buf, format='JPEG')
encoded_image = buf.getvalue()
print(encoded_image)

# get name
bounding_box = img_name.split('-')[2]
xmin, ymin = tuple(bounding_box.split('_')[0].split('&'))
xmax, ymax = tuple(bounding_box.split('_')[1].split('&'))

xmax = int(xmax)
xmin = int(xmin)
ymax = int(ymax)
ymin = int(ymin)


# random crop
top = max(0, ymax-720)
bottom = min(1163-720, ymin)
crop_top = np.random.randint(top, bottom)

ymax = ymax-crop_top
ymin = ymin-crop_top
image = image.crop((0, crop_top, 720, crop_top+720))
image1 = ImageDraw.Draw(image)
image1.rectangle([xmin, ymin, xmax, ymax])

width, height = image.size
#image.show("test")
