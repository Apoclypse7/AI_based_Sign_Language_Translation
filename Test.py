import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3
import time
import threading

# Initialize pyttsx3 engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300
counter = 0
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

def ttsx(max_prediction):
    time.sleep(0)
    engine.say(max_prediction)
    engine.runAndWait()
    # t1.kill()

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        if imgCrop.size > 0:  # Ensure imgCrop has valid dimensions
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Get prediction from the classifier
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print("Prediction:", prediction, "Index:", index)

            # Find the index of the maximum probability
            max_prob_index = np.argmax(prediction)
            max_prediction = labels[max_prob_index]

            # Speak out the maximum prediction
            t1 = threading.Thread(target=ttsx, args = (max_prediction,))
            t1.start()
            
            # Draw bounding box and label on the output image
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)


    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
