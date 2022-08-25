import matplotlib
matplotlib.use("Agg")

from config import emotion_config as config
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.callbacks.epochceackpoint import EpochCheckpoint
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from pyimagesearch.io.hdf5datagenerateor import Hdf5DatasetGenerator
from pyimagesearch.nn.conv.emotionvggnet import EmotionVGGNet
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.models import load_model
import keras.backend as k
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-c","--checkpoints",required=True,help="path to output dirc")
ap.add_argument("-m","--model",type=str,help ="path to specific model to load")
ap.add_argument("-s","--start_epoch",type=int,default=0)
args = vars(ap.parse_args())

trainAug = ImageDataGenerator(rotation_range=10,zoom_range=0.1,horizontal_flip=True,rescale=1/255.0,fill_mode="nearest")
valAug = ImageDataGenerator(rescale=1/255.0)
iap = ImageToArrayPreprocessor()

trainGen = Hdf5DatasetGenerator(config.TRAIN_HDF5,config.BATCH_SIZE,aug=trainAug,preprocessors=[iap],classes=config.NUM_CLASSES)
valGen = Hdf5DatasetGenerator(config.VAL_HDF5,config.BATCH_SIZE,aug=valAug,preprocessors=[iap],classes=config.NUM_CLASSES)

if args["model"] is None:
    print("[INFO] compiling model...")
    model = EmotionVGGNet.build(48,48,1,config.NUM_CLASSES)
    opt = tf.keras.optimizers.Adam(lr=1e-3)
    model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

else:
    print("[INFO] loading model {}...".format(args["model"]))
    model = load_model(args["model"])

    print("[INFO] old learning rate = {}".format(k.get_value(model.optimizer.lr)))

    k.set_value(model.optimizer.lr,5e-5)
    print("[INFO] new learning rate = {}".format(k.get_value(model.optimizer.lr)))

figPath = os.path.sep.join([config.OUTPUT_PATH,"vggnet_emotion.png"])
jsonPath = os.path.sep.join([config.OUTPUT_PATH,"vggnet_emotion.json"])
callbacks = [EpochCheckpoint(args["checkpoints"],every=5,startAt=args["start_epoch"]),TrainingMonitor(figPath,jsonPath,args["start_epoch"])]

model.fit(trainGen.generator(),steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,validation_data=valGen.generator()
,validation_steps= valGen.numImages // config.BATCH_SIZE,epochs=80,max_queue_size=config.BATCH_SIZE*2,callbacks=callbacks,verbose=1)

trainGen.close()
valGen.close()