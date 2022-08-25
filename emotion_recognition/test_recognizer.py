from config import emotion_config as config
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.io.hdf5datagenerateor import Hdf5DatasetGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m","--model",type=str,help="path to model")
args = vars(ap.parse_args())

testAug = ImageDataGenerator(rescale=1/255.0)
iap = ImageToArrayPreprocessor()

testGen = Hdf5DatasetGenerator(config.TEST_HDF5,config.BATCH_SIZE,aug=testAug,preprocessors=[iap],classes=config.NUM_CLASSES)
print("[INFO] loading model {}...".format(args["model"]))
model = load_model(args["model"])

(loss,acc) = model.evaluate(testGen.generator(),steps=testGen.numImages // config.BATCH_SIZE,max_queue_size=config.BATCH_SIZE * 2)
print("[INFO] accuracy : {:.2f}".format(acc*100))
testGen.close()