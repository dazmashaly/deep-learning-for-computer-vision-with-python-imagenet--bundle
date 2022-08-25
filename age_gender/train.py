from config import age_gender_config as config
from pyimagesearch.nn.mxconv.mxagegender import MxAgeGender
from pyimagesearch.nn.mxconv.mxsqueezenet import MxSqueezeNet
from pyimagesearch.utils.agegenderhelper import AgeGenderHelper
from pyimagesearch.mxcallbacks.mxmetrices import one_off_callback
import mxnet as mx 
import argparse
import logging
import json
import os
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-c","--checkpoints",required=True,help="path to output cheakpoint")
ap.add_argument("-p","--prefix",required=True,help="name of model prefix")
ap.add_argument("-s","--start_epoch",type=int,default=0)
args = vars(ap.parse_args())

logging.basicConfig(level=logging.DEBUG,filename="training_{}.log".format(args["start_epoch"]),filemode="w")

means = json.loads(open(config.DATASET_MEAN).read())
batchSize = config.BATCH_SIZE

trainIter = mx.io.ImageRecordIter(path_imgrec=config.TRAIN_MX_REC,data_shape=(3,227,227),batch_size=batchSize,rand_crop=True,
rand_mirror=True,rotate=7,mean_r=means["R"],mean_g=means["G"],mean_b=means["b"],preprocess_threads=2)

valIter = mx.io.ImageRecordIter(path_imgrec=config.VAL_MX_REC,data_shape=(3,227,227),batch_size=batchSize,mean_r=means["R"],mean_g=means["G"],mean_b=means["b"])

opt = mx.optimizer.Adam(learning_rate=1e-4,wd=0.0005)

checkpointsPath = os.path.sep.join([args["checkpoints"],args["prefix"]])
argParams = None
auxParams = None

if args["start_epoch"] <=0:
    print("[INFO] building network....")
    model = MxAgeGender.build(config.NUM_CLASSES)

else:
    print("[INFO] loading model epoch {}....".format(args["start_epoch"]))
    model = mx.model.FeedForward.load(checkpointsPath,args["start_epoch"])
    argParams = model.arg_params
    auxParams = model. aux_params
    model = model.symbol

model = mx.model.FeedForward(ctx=[mx.gpu(0)],symbol=model,initializer=mx.initializer.Xavier(),arg_params=argParams,
aux_params=auxParams,num_epoch=150,begin_epoch=args["start_epoch"])

le = pickle.loads(open(config.LABEL_ENCODER_PATH,"rb").read())
agh = AgeGenderHelper(config)
oneOff = agh.buildOneOffMappings(le)

batchEndCBs = [mx.callback.Speedometer(batchSize,30)]
epochEndCBs = [mx.callback.do_checkpoint(checkpointsPath)]
metries =[mx.metric.Accuracy(),mx.metric.CrossEntropy()]



print("[INFO] training the network....")
model.fit(X=trainIter,eval_data=valIter,eval_metric=metries,
batch_end_callback=batchEndCBs,epoch_end_callback=epochEndCBs)


