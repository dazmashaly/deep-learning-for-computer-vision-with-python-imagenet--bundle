from ntpath import join
from os import path

BASE_PATH = "D:\Data"

IMAGES_PATH = BASE_PATH
DEVKIT_PATH = "D:\Data\divkit\data"
WORD_IDS = path.sep.join([DEVKIT_PATH,"map_clsloc.txt"])
TRAIN_LIST = path.sep.join([DEVKIT_PATH,"train_cls.txt"])
VAL_LIST =  path.sep.join([DEVKIT_PATH,"val.txt"])
VAL_LABELS =  path.sep.join([DEVKIT_PATH,"ILSVRC2015_clsloc_validation_ground_truth.txt"])
VAL_BLACKLIST = path.sep.join([DEVKIT_PATH,"ILSVRC2015_clsloc_validation_blacklist.txt"])
NUM_CLASSES = 1000
NUM_TEST_IMAGES = 50000
MX_OUTPUT = "D:\Data\mxout"
TRAIN_MX_LIST = path.sep.join([MX_OUTPUT,"lists\\train.lst"])
VAL_MX_LIST = path.sep.join([MX_OUTPUT,"lists\\val.lst"])
TEST_MX_LIST = path.sep.join([MX_OUTPUT,"lists\\test.lst"])

TRAIN_MX_REC = path.sep.join([MX_OUTPUT,"rec\\train.rec"])
VAL_MX_REC = path.sep.join([MX_OUTPUT,"rec\\val.rec"])
TEST_MX_REC = path.sep.join([MX_OUTPUT,"rec\\test.rec"])

DATASET_MEAN = "D:\\projeckts\\deeplearning\\work\\DL_for_CV\\img_bun\\alexnet_mx\\output\\imagenet_mean.json"
BATCH_SIZE = 64
NUM_DEVICE = 1