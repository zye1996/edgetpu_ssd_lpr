import sys
import time

def SpeedTest(image_path):
    grr = Image.open(image_path)
    # model = pr.LPR("model/cascade.xml","model/model12.h5","model/ocr_plate_all_w_rnn_2.h5")
    model = pr.LPR("export_graph/edgetpu_models/detection_model_edgetpu.tflite",
                   "export_graph/model12.h5",
                   "export_graph/edgetpu_models/ocr_model_edgetpu.tflite", cnn=True, tpu=True)
    model.SimpleRecognizePlateByE2E(grr)
    t0 = time.time()
    for x in range(20):
        model.SimpleRecognizePlateByE2E(grr)
    t = (time.time() - t0) / 20.0
    print("Image size :" + str(grr.size[1])+"x"+str(grr.size[0]) +  " need " + str(round(t*1000,2))+"ms")


from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
fontC = ImageFont.truetype("./Font/platech.ttf", 14, 0)

def drawRectBox(image,rect,addText):
    cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0,0, 255), 2, cv2.LINE_AA)
    cv2.rectangle(image, (int(rect[0]-1), int(rect[1])-16), (int(rect[0] + 115), int(rect[1])), (0, 0, 255), -1,
                  cv2.LINE_AA)
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.text((int(rect[0]+1), int(rect[1]-16)), addText, (255, 255, 255), font=fontC)
    imagex = np.array(img)
    return imagex


import HyperLPRLite as pr
import cv2
import numpy as np

SpeedTest("images_rec/2_resized.jpg")

img = Image.open("images_rec/1_resized.jpg")
model = pr.LPR("export_graph/edgetpu_models/detection_model_edgetpu.tflite",
               "export_graph/model12.h5",
               "export_graph/edgetpu_models/ocr_model_edgetpu.tflite", cnn=True, tpu=True)
for pstr, confidence, rect in model.SimpleRecognizePlateByE2E(img):
    #if confidence>0.1:
    image = drawRectBox(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), rect, pstr+" "+str(round(confidence,3)))
    print("plate_str:")
    print(pstr)
    print("plate_confidence")
    print(confidence)
            
cv2.imshow("image", image)
cv2.waitKey(0)

cv2.destroyAllWindows()




