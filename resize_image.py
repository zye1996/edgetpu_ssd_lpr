import cv2
import os

SQUARE = True
IMG_NAME = "14.png"
W = 720
H = 1163

if __name__ == "__main__":


    img = cv2.imread(os.path.join("images_rec", IMG_NAME))
    h, w, c = img.shape

    if SQUARE:
        if w > h:
            img_resized = img[:, (w-h)//2:(w+h)//2]
        else:
            img_resized = img[(h-w)//2:(h+w)//2, :]

    else:
        w_target = w
        h_target = int(w * H / W)
        if h_target > h:
            h_target = h
            w_target = int(h * W / H)
            img_resized = img[:, (w-w_target)//2:(w+w_target)//2]
        else:
            img_resized = img[(h-h_target)//2:(h+h_target)//2,:]

    cv2.imwrite(os.path.join("images_rec", IMG_NAME.split('.')[0]+'_resized.jpg'),
                img_resized)
