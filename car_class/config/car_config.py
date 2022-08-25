from os import path

BASE_PATH = "D:\\projeckts\\deeplearning\\work\\DL_for_CV\\img_bun\\cars"

TRAIN_IMAGES_PATH = path.sep.join([BASE_PATH,"cars_train"])

TEST_IMAGES_PATH = path.sep.join([BASE_PATH,"cars_test"])
MX_OUTPUT  = path.sep.join([BASE_PATH,"output"])
TRAIN_MX_LIST = path.sep.join([BASE_PATH,"\\lists\\train.lst"])
VAL_MX_LIST = path.sep.join([BASE_PATH,"\\lists\\cal.lst"])
TEST_MX_LIST = path.sep.join([BASE_PATH,"\\lists\\test.lst"])
TRAIN_MX_REC = path.sep.join([BASE_PATH,"\\rec\\train.rec"])
VAL_MX_REC = path.sep.join([BASE_PATH,"\\rec\\cal.rec"])
TEST_MX_REC = path.sep.join([BASE_PATH,"\\rec\\test.rec"])
LABEL_ENCODER = path.sep.join([BASE_PATH,"\\output\\le.cpickle"])
R_MEAN = 123.68
G_MEAN = 116.779
B_MEAN = 103.939

NUM_CLASSES = 196
NUM_VAL = 0.15
NUM_TEST = 0.15
BATCH_SIZE = 4
