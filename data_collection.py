import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os
from datetime import datetime

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
folder = 'Data/L'
counter = 0

if not os.path.exists(folder):
    os.makedirs(folder)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        if imgCrop.size != 0:
            imgCropShape = imgCrop.shape
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                # Create a blank canvas of imgSize
                imgCanvas = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                # Paste the resized image onto the canvas
                imgCanvas[:, :wCal] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                # Create a blank canvas of imgSize
                imgCanvas = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                # Paste the resized image onto the canvas
                imgCanvas[:hCal, :] = imgResize
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgCanvas)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord("s"):
        counter += 1
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(f'{folder}/Image_{current_time}_{counter}.jpg', imgCanvas)
        print(f"Image saved: Image_{current_time}_{counter}.jpg")

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
