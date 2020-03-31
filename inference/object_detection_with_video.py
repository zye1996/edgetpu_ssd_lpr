import cv2

cap = cv2.VideoCapture('image_rec/test_video.mp4')

if (cap.isOpened()== False):
    print("Error opening video stream or file")

while cap.isOpened():
    ret, image = cap.read()
    cv2.imshow('video', image)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()




