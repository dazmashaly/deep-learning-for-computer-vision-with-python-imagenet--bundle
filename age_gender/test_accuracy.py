from pyimagesearch.utils.agegenderhelp import AgeGenderHelper
from config import age_gender_config as config
from pyimagesearch.mxcallbacks.mxmetrices import _compute_one_off
from config import age_gender_deploy as deploy
import cv2
from mxnet import nd
import mxnet as mx
import argparse
import pickle 
import json
import os

ageLE = pickle.loads(open(deploy.AGE_LABEL_ENCODER,"rb").read())

ap = argparse.ArgumentParser()
ap.add_argument("-c","--checkpoints",required=True,help="path to checkpoint")
ap.add_argument("-p","--prefix",required=True,help="name of model ")
ap.add_argument("-e","--epoch",type=int,required=True,help="number of epoch")
args = vars(ap.parse_args())
means = json.loads(open(config.DATASET_MEAN).read())

testIter =mx.io.ImageRecordIter(path_imgrec=config.TEST_MX_REC,data_shape=(3,227,227),
batch_size=config.BATCH_SIZE,mean_r=means["R"],mean_g=means["G"],mean_b=means["b"])

print("[INFO] loading model...")
checkpointsPath = os.path.sep.join([args["checkpoints"],args["prefix"]])
model = mx.model.FeedForward.load(checkpointsPath,args["epoch"])
model = mx.model.FeedForward(ctx=[mx.gpu(0)],symbol=model.symbol,arg_params=model.arg_params,aux_params=model.aux_params)

print("[INFO] predicting on {} test data".format(config.DATASET_TYPE))
metricss = [mx.metric.Accuracy()]
acc = model.score(testIter,eval_metric=metricss)

print("[INFO] rank-1: {:.2f}%".format(acc[0]*100))


if config.DATASET_TYPE == "age":
    arg = model.arg_params
    aux = model.aux_params
    model = mx.mod.Module(symbol=model.symbol,context=[mx.gpu(0)])
    model.bind(data_shapes=testIter.provide_data,label_shapes=testIter.provide_label)
    model.set_params(arg,aux)
    le = pickle.loads(open(config.LABEL_ENCODER_PATH,"rb").read())
    agh = AgeGenderHelper(config)
    oneoff =agh.buildOneOffMappings(le)
    acc = _compute_one_off(model,testIter,oneoff)
    print("[INFO] one off: {:.2f}".format(acc*100))