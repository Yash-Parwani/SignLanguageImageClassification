import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
from gtts import gTTS
import os
from pydub import AudioSegment
from pydub.playback import play

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

folder = "Data/hello"
counter = 0

labels = ["A", "am", "B", "C", "GoodBye", "hello", "I", "I Love You", "me", "no", "Thank You", "yes"]

confidence_threshold = 0.5  # Adjust this threshold as needed

prediction_history = []
last_prediction_time = time.time()

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w+offset, y + h+offset), (255, 0, 255), 4)

        # Store the prediction and confidence
        prediction_history.append((labels[index], prediction))
        current_time = time.time()

        # Check if 3 seconds have passed and choose the label with the highest confidence
        if current_time - last_prediction_time >= 3:
            if prediction_history:
                best_prediction = max(prediction_history, key=lambda x: x[1])
                text_to_speech = best_prediction[0]

                # Convert the text to speech using gTTS
                tts = gTTS(text_to_speech)
                tts.save("output.mp3")

                # Play the generated speech using pydub
                sound = AudioSegment.from_mp3("output.mp3")
                play(sound)

                # Clear the prediction history and update the last prediction time
                prediction_history.clear()
                last_prediction_time = current_time

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
