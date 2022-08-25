import re
from imutils import paths
from matplotlib import widgets
import numpy as np
import progressbar
import argparse
import imutils
import random 
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to dataset")
ap.add_argument("-o","--output",required=True,help="path to output")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["dataset"]))[:10000]
random.shuffle(imagePaths)

angles ={}
widgets = ["Building Dataset: ",progressbar.Percentage()," ",progressbar.Bar()," ",progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths),widgets=widgets).start()

for (i,imagepath) in enumerate(imagePaths):

    angle = np.random.choice([0,90,180,270])
    image = cv2.imread(imagepath)
    if image is None:
        continue
    image = imutils.rotate_bound(image,angle)
    base = os.path.sep.join([args["output"],str(angle)])

    if not os.path.exists(base):
        os.makedirs(base)

    #extention
    ext = imagepath[imagepath.rfind("."):]
    outputPath = [base,"image_{}_{}".format(str(angles.get(angle,0)).zfill(5),ext)]
    outputPath = os.path.sep.join(outputPath)
    cv2.imwrite(outputPath,image)
    c = angles.get(angle,0)
    angles[angle] =c+1
    pbar.update(i)

pbar.finish()

for angle in sorted(angles.keys()):
    print("[INFO] angle={}: {:,}".format(angle,angles[angle]))
