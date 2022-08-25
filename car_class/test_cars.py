from config import car_config as config
from pyimagesearch.utils.ranked import rank5_accuracy
import mxnet as mx
import argparse
import pickle
import os 
import logging

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
help="path to output checkpoint directory")
ap.add_argument("-p", "--prefix", required=True,
 help="name of model prefix")
ap.add_argument("-e", "--epoch", type=int, required=True, help="epoch # to load")
args = vars(ap.parse_args())

testIter = mx.io.ImageRecordIter(path_imgrec=config.TEST_MX_REC,data_shape=(3, 224, 224),batch_size=config.BATCH_SIZE,
mean_r=config.R_MEAN,mean_g=config.G_MEAN,mean_b=config.B_MEAN)

print("[INFO] loading pre-trained model...")
checkpointsPath = os.path.sep.join([args["checkpoints"],args["prefix"]])
(symbol, argParams, auxParams) = mx.model.load_checkpoint(checkpointsPath, args["epoch"])

# construct the model
model = mx.mod.Module(symbol=symbol, context=[mx.gpu(0)])
model.bind(data_shapes=testIter.provide_data,label_shapes=testIter.provide_label)
model.set_params(argParams, auxParams)

print("[INFO] evaluating model...")
predictions = []
targets = []

# loop over the predictions in batches
for (preds, _, batch) in model.iter_predict(testIter):
# convert the batch of predictions and labels to NumPy
# arrays
    preds = preds[0].asnumpy()
    labels = batch.label[0].asnumpy().astype("int")

# update the predictions and targets lists, respectively
    predictions.extend(preds)
    targets.extend(labels)

targets = targets[:len(predictions)]
(rank1, rank5) = rank5_accuracy(predictions, targets)
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
print("[INFO] rank-5: {:.2f}%".format(rank5 * 100))