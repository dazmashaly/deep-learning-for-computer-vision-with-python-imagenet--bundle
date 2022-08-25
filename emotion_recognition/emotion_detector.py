from chardet import detect
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
from time import sleep
import cv2
i = 0
ap = argparse.ArgumentParser()
ap.add_argument("-c","--cascade",required=True,help="path to where the face cascade is")
ap.add_argument("-m","--model" ,required= True)
ap.add_argument("-v","--video")
args = vars(ap.parse_args())
detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])
EMOTIONS = ["angry", "scared", "happy", "sad", "surprised","neutral"]
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])
classLabels = ["sad","happy"]
while True:
    #grab the current frame
    (grabbed, frame) = camera.read()

    #if no frame was grabed the we reached the end
    if args.get("video") and not grabbed:
        break

    #resize the frame ,grayscale it , and clone it to draw on it later
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = np.zeros((220,300,3),dtype="uint8")
    frameClone = frame.copy()
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)

    if len(rects) >0:
        rect = sorted (rects,reverse=True,key= lambda x: (x[2] - x[0] ) * (x[3] - x[1])) [0]
        (fx,fy,fw,fh) = rect

        roi = gray[fy:fy + fh, fx:fx + fw]
        roi = cv2.resize(roi,(48,48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi,axis=0)

        preds = model.predict(roi)[0]
        label = EMOTIONS[preds.argmax()]

        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            text = "{}: {:.2f}%".format(emotion,prob*100)

            w = int(prob * 300)
            cv2.rectangle(canvas, (5, (i * 35) + 5),
            (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255, 255, 255), 2)
        cv2.putText(frameClone, label, (fx, fy - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fx, fy), (fx + fw, fy + fh),
        (0, 0, 255), 2)
    cv2.imshow("Face", frameClone)
    cv2.imshow("Probabilities", canvas)

# if the ’q’ key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()