import os
from os import path
DATASET_TYPE ="age"

BASE_PATH = "D:\Data\gen_rec"
OUTPUT_BASE = "D:\Data\gen_rec\output"
MX_OUTPUT = BASE_PATH

IMAGES_PATH = path.sep.join([BASE_PATH,"aligned"])
LABELS_PATH = path.sep.join([BASE_PATH,"folds"])

NUM_VAL = 0.15
NUM_TEST = 0.15

BATCH_SIZE = 64
NUM_DEVICES = 1

if DATASET_TYPE == "age":
    NUM_CLASSES = 8
    LABEL_ENCODER_PATH = path.sep.join([OUTPUT_BASE,"age_le.cpickle"])

    TRAIN_MX_LIST = path.sep.join([MX_OUTPUT,"list\\age_train.lst"])
    VAL_MX_LIST = path.sep.join([MX_OUTPUT,"list\\age_val.lst"])
    TEST_MX_LIST = path.sep.join([MX_OUTPUT,"list\\age_test.lst"])
    TRAIN_MX_REC = path.sep.join([MX_OUTPUT,"rec\\age_train.rec"])
    VAL_MX_REC = path.sep.join([MX_OUTPUT,"rec\\age_val.rec"])
    TEST_MX_REC = path.sep.join([MX_OUTPUT,"rec\\age_test.rec"])

    DATASET_MEAN = path.sep.join([OUTPUT_BASE,"age_adience_mean.json"])

elif DATASET_TYPE == "gender":
    NUM_CLASSES = 2
    LABEL_ENCODER_PATH = path.sep.join([OUTPUT_BASE,"gender_le.cpickle"])

    TRAIN_MX_LIST = path.sep.join([MX_OUTPUT,"list\\gender_train.lst"])
    VAL_MX_LIST = path.sep.join([MX_OUTPUT,"list\\gender_val.lst"])
    TEST_MX_LIST = path.sep.join([MX_OUTPUT,"list\\gender_test.lst"])
    TRAIN_MX_REC = path.sep.join([MX_OUTPUT,"rec\\gender_train.rec"])
    VAL_MX_REC = path.sep.join([MX_OUTPUT,"rec\\gender_val.rec"])
    TEST_MX_REC = path.sep.join([MX_OUTPUT,"rec\\gender_test.rec"])

    DATASET_MEAN = path.sep.join([OUTPUT_BASE,"gender_adience_mean.json"])
