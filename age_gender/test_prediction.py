from genericpath import isdir
import cv2
from config import age_gender_deploy as deploy
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.meanPreprocessor import MeanPreprocessor
from pyimagesearch.preprocessing.croppreprocessor import CropPreprocessor
from pyimagesearch.utils.agegenderhelp import AgeGenderHelper
from imutils.face_utils import FaceAligner
from imutils import paths
from imutils import face_utils
import numpy as np
import mxnet as mx
import argparse
import pickle
import dlib
import imutils
import json
import difflib
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to input image,images")
args = vars(ap.parse_args())
#could be path to image or dirctory of images

print("[INFO] loading label encodersand mean files...")
ageLE = pickle.loads(open(deploy.AGE_LABEL_ENCODER,"rb").read())
genderLE = pickle.loads(open(deploy.GENDER_LABEL_ENCODER,"rb").read())
ageMeans = json.loads(open(deploy.AGE_MEANS).read())
genderMeans = json.loads(open(deploy.GENDER_MEANS).read())

print("[INFO] loading model...")
agePath = os.path.sep.join([deploy.AGE_NETWORK_PATH,deploy.AGE_PREFIX])
genderPath = os.path.sep.join([deploy.GENDER_NETWORK_PATH,deploy.GENDER_PREFIX])
ageModel = mx.model.FeedForward.load(agePath,deploy.AGE_EPOCH)
genderModel = mx.model.FeedForward.load(genderPath,deploy.GENDER_EPOCH)

print("[INFO] compiling models...")
ageModel = mx.model.FeedForward(ctx=[mx.gpu(0)],symbol=ageModel.symbol,arg_params=ageModel.arg_params
,aux_params=ageModel.aux_params)
genderodel = mx.model.FeedForward(ctx=[mx.gpu(0)],symbol=genderModel.symbol,arg_params=genderModel.arg_params
,aux_params=genderModel.aux_params)

sp = SimplePreprocessor(width=256,height=256,inter=cv2.INTER_CUBIC)
cp = CropPreprocessor(width=227,height=227,horiz=True)
ageMP = MeanPreprocessor(ageMeans["R"],ageMeans["G"],ageMeans["b"])
genderMP = MeanPreprocessor(genderMeans["R"],genderMeans["G"],genderMeans["b"])
iap = ImageToArrayPreprocessor(dataFormat="channels_first")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(deploy.DLIB_LANDMARK_PATH)
fa = FaceAligner(predictor)

imagePaths = [args["image"]]
if os.path.isdir(args["image"]):
    imagePaths = sorted(list(paths.list_files(args["image"])))
for imagePath in imagePaths:
    print("[INFO] processing {}".format(imagePath))
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #detect faces in the grayscale image
    rects = detector(gray,1)

    for rect in rects:
        #determine the facial landmarks and align the face
        shape = predictor(gray,rect)
        face = fa.align(image, gray, rect)
        face = sp.Preprocess(face)
        patches = cp.Preprocess(face)
        agePatches = np.zeros((patches.shape[0],3,227,227),dtype="float")
        genderPatches = np.zeros((patches.shape[0],3,227,227),dtype="float")
        for j in np.arange(0,patches.shape[0]):
            agePatch = ageMP.Preprocess(patches[j])
            genderPatch = genderMP.Preprocess(patches[j])
            agePatch = iap.Preprocess(agePatch)
            genderPatch = iap.Preprocess(genderPatch)

            agePatches[j] = agePatch
            genderPatches[j] = genderPatch
        
        agePreds = ageModel.predict(agePatches)
        genderPreds = genderModel.predict(genderPatches)
        agePreds =agePreds.mean(axis=0)
        genderPreds =genderPreds.mean(axis=0)

        ageCanvas = AgeGenderHelper.visualizeAge(agePreds,ageLE)
        genderCanvas = AgeGenderHelper.visualizeGender(genderPreds,genderLE)

        clone = image.copy()
        (x,y,w,h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(clone,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.imshow("INPUT",clone)
        cv2.imshow("FACE",face)
        cv2.imshow("age preds",ageCanvas)
        cv2.imshow("gender preds",genderCanvas)
        cv2.waitKey(0)