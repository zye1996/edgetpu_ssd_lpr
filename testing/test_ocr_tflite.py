import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# MODEL = '../export_graph/ocr_models/ocr_model_quant_int8.tflite'
MODEL = '../export_graph/edgetpu_models/ocr_model_edgetpu.tflite'

chars = [u"京", u"沪", u"津", u"渝", u"冀", u"晋", u"蒙", u"辽", u"吉", u"黑", u"苏", u"浙", u"皖", u"闽", u"赣", u"鲁", u"豫", u"鄂", u"湘", u"粤", u"桂",
         u"琼", u"川", u"贵", u"云", u"藏", u"陕", u"甘", u"青", u"宁", u"新", u"0", u"1", u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9", u"A",
         u"B", u"C", u"D", u"E", u"F", u"G", u"H", u"J", u"K", u"L", u"M", u"N", u"P", u"Q", u"R", u"S", u"T", u"U", u"V", u"W", u"X",
         u"Y", u"Z",u"港",u"学",u"使",u"警",u"澳",u"挂",u"军",u"北",u"南",u"广",u"沈",u"兰",u"成",u"济",u"海",u"民",u"航",u"空"
         ]

def fastdecode(y_pred):
    results = ""
    confidence = 0.0
    table_pred = y_pred.reshape(-1, len(chars)+1)
    print(table_pred.shape)
    res = table_pred.argmax(axis=1)
    for i,one in enumerate(res):
        if one < len(chars) and (i==0 or (one!=res[i-1])):
            results+= chars[one]
            confidence+=table_pred[i][one]
    confidence/= len(results)
    return results, confidence


# set interpreter
model_file, *device = MODEL.split('@')
interpreter = tflite.Interpreter(MODEL, experimental_delegates=[
    tflite.load_delegate('libedgetpu.so.1',
                         {'device': device[0]} if device else {})
])
interpreter.allocate_tensors()

# get i/o tensor
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']
mean, std_dev = interpreter.get_output_details()[0]['quantization']

# read image
img = plt.imread('../images_rec/ocr_images/2.jpg')
img = cv2.resize(img, (160, 40))
img = img.transpose(1, 0, 2) #.astype(np.float32)
img = img[np.newaxis, :, :, :]

# feed forward
interpreter.set_tensor(input_index, img)
interpreter.invoke()
result = interpreter.get_tensor(output_index)
result = (result - mean) / std_dev

# decode
result = result[:,2:,:]
print(fastdecode(result))

