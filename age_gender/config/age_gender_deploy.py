from os import path
OUTPUT_BASE = "D:\Data\gen_rec\output"

DLIB_LANDMARK_PATH = "D:\Data\shape_predictor_68_face_landmarks.dat"

AGE_NETWORK_PATH = "D:\Data\gen_rec\checkpoints"
AGE_PREFIX = "alex_Age"
AGE_EPOCH = 118
AGE_LABEL_ENCODER = path.sep.join([OUTPUT_BASE,"age_le.cpickle"])
AGE_MEANS = path.sep.join([OUTPUT_BASE,"age_adience_mean.json"])

GENDER_NETWORK_PATH = "D:\Data\gen_rec\checkpoints"
GENDER_PREFIX = "gender"
GENDER_EPOCH = 150
GENDER_LABEL_ENCODER = path.sep.join([OUTPUT_BASE,"gender_le.cpickle"])
GENDER_MEANS = path.sep.join([OUTPUT_BASE,"gender_adience_mean.json"])