from ast import arg
import cv2
from config import age_gender_config as config
from config import age_gender_deploy as deploy
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.meanPreprocessor import MeanPreprocessor
from pyimagesearch.utils.agegenderhelp import AgeGenderHelper
import numpy as np 
import mxnet as mx
import argparse
import pickle
import json
import os
import imutils
ap = argparse.ArgumentParser()
ap.add_argument("-s","--sample_size",type=int,default=10,help="number of images to load")
args= vars(ap.parse_args())

print("[INFO] loading label encoders and mean files...")
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

sp =SimplePreprocessor(width=227,height=227,inter=cv2.INTER_CUBIC)
ageMP=MeanPreprocessor(ageMeans["R"],ageMeans["G"],ageMeans["b"])
genderMP=MeanPreprocessor(genderMeans["R"],genderMeans["G"],genderMeans["b"])
iap = ImageToArrayPreprocessor(dataFormat="channels_first")

rows = open(config.TEST_MX_LIST).read().strip().split("\n")
rows = np.random.choice(rows,size=args["sample_size"])

for row in rows:
    (_,gtLabel,imagePath) = row.strip().split("\t")
    image = cv2.imread(imagePath)
    ageImage = iap.Preprocess(ageMP.Preprocess(sp.Preprocess(image)))
    genderImage = iap.Preprocess(genderMP.Preprocess(sp.Preprocess(image)))

    ageImage = np.expand_dims(ageImage,axis=0)
    genderImage = np.expand_dims(genderImage,axis=0)

    agePreds = ageModel.predict(ageImage)[0]
    genderPreds = genderModel.predict(genderImage)[0]

    argIdx = np.argsort(agePreds)[::-1]
    genderIdx = np.argsort(genderPreds)[::-1]

    ageCanvas = AgeGenderHelper.visualizeAge(preds=argIdx,le=ageLE)
    image = imutils.resize(image,width=400)

    
    gtLabel = ageLE.inverse_transform(np.array([int(gtLabel)]))
    stlabel = ageLE.inverse_transform(argIdx)
    print(stlabel)
    text = "Actual: {}-{}".format(*gtLabel[0].split("_"))
    cv2.putText(image,text,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),3)
    cv2.imshow("image",image)
    cv2.imshow("Age probs",ageCanvas)
    cv2.waitKey(0)