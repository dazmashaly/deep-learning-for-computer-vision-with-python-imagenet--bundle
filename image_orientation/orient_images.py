from keras.applications.vgg16 import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from imutils import paths
from matplotlib import image
import pickle
import numpy as np 
import progressbar
import argparse
import random
import os
import imutils
import h5py
import cv2
ap = argparse.ArgumentParser()
ap.add_argument("-d","--db", required=True,help="path to hdf5")
ap.add_argument("-i","--dataset",required=True)
ap.add_argument("-m","--model",required=True,help="path to trained model")
args = vars(ap.parse_args())

db = h5py.File(args["db"])
labelNames = [int(angle) for angle in db["label_names"][:]]
db.close()

print("[INFO] sampling images...")

imagePaths = list(paths.list_images(args["dataset"]))
imagePaths = np.random.choice(imagePaths,size=(10,),replace=False)

print("[INFO] loading network...")
vgg = VGG16(include_top=False,weights="imagenet")
print("[INFO] loading model")
model = pickle.loads(open(args["model"],"rb").read())

for imagePath in imagePaths:
    orig = cv2.imread(imagePath)

    image = load_img(imagePath,target_size=(224,224))
    image = img_to_array(image)
    image = np.expand_dims(image,axis=0)
    image = imagenet_utils.preprocess_input(image)
    features =  vgg.predict(image)
    features = features.reshape((features.shape[0],512 * 7 * 7))

    angle = model.predict(features)
    angle = labelNames[angle[0]]

    rotated = imutils.rotate_bound(orig,360-angle)

    cv2.imshow("original",orig)
    cv2.imshow("fixed",rotated)
    cv2.waitKey(0)