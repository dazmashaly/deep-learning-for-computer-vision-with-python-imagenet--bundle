# Adrian Rosebrock - Deep Learning for Computer Vision with Python imagenet Bundle(2018, PyImageSearch)

**pdf version attached**

https://www.mediafire.com/file/lmya0c3ah1bpzgu/Adrian_Rosebrock_-_Deep_Learning_for_Computer_Vision_with_Python_imagenet_Bundle%25282017%252C_PyImageSearch%2529.pdf/file


# datasets used in this book:

imagenet: https://image-net.org/challenges/LSVRC/2012/2012-downloads.php

only download the training (138 GB) data and the validation (7 GB),
when you download the data you wont find the files trian_cls.txt and val.txt 
so i uploaded them with the divket in here : https://www.mediafire.com/file/nvp94jinz1hw2vg/divkit.rar/file

fer2013: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=fer2013.tar.gz

indoor cvpr: http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar

cars: http://ai.stanford.edu/~jkrause/car196/cars_train.tgz http://ai.stanford.edu/~jkrause/car196/cars_test.tgz

for the divkit https://www.mediafire.com/file/cfc2oq5foix6jj6/devkit.rar/file

the vgg parameters http://data.mxnet.io.s3-website-us-west-1.amazonaws.com/models/imagenet/vgg/vgg16-0000.params

http://data.mxnet.io.s3-website-us-west-1.amazonaws.com/models/imagenet/vgg/vgg16-symbol.json

adience aligned : http://www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/aligned.tar.gz

and the folds : https://www.mediafire.com/file/dup40uy6df3vb6r/folds.rar/file

the data for chapters 16:18 (r_cnn and ssd) are no longer avilable and i was not able to find them.
some of the data or code used to build the data may be diffrent from the one in the book.


#-------

when building the imagenet after installing mxnet open the folder you installed it in and add this tools folder
https://www.mediafire.com/file/4pfai8fzhfev8sq/tools.rar/file inside you would find the im2rec.py file you need.
open the file inside the tools folder and run the commands in the arguments.

#-------

make sure to change the paths in the config files to your own.

some filenames may not exactliy match the ones in the book but they are the same

#------

# enviorment

all code that does not enclude mxnet in here was made using python 3.10 ,cuda 11.2 and the following packages virsons
same as the previous books 

but all code the needed mxnet was made using python 3.6.8 and  cuda 10 and the following packages:

dlib                         19.24.0

glmnet-py                    0.1.0b2

gluoncv                      0.10.5.post0

google-auth                  1.35.0

google-auth-oauthlib         0.4.6

google-pasta                 0.2.0

graphviz                     0.8.4

grpcio                       1.44.0

h5py                         3.1.0

importlib-metadata           4.8.3

importlib-resources          5.4.0

imutils                      0.5.4

jupyter-client               7.1.0

jupyter-core                 4.9.1

keras                        2.6.0

Keras-Preprocessing          1.1.2

matplotlib                   3.3.4

matplotlib-inline            0.1.3

mlxtend                      0.19.0

mxnet                        1.8.0

mxnet-cu100                  1.5.0

numpy                        1.19.5

nvidia-ml-py3                7.352.0

nvidia-smi                   0.1.3

opencv-python                4.5.5.64

pandas                       1.3.4

pip                          21.3.1

progressbar                  2.5

requests                     2.27.1

requests-oauthlib            1.3.1

rsa                          4.8

scikit-image                 0.19.2

scikit-learn                 0.24.2

scipy                        1.5.4

seaborn                      0.11.2

setuptools                   57.4.0

setuptools-scm               6.3.2

six                          1.15.0

sklearn                      0.0

statsmodels                  0.13.1

tensorboard                  2.6.0

tensorboard-data-server      0.6.1

tensorboard-plugin-wit       1.8.1

tensorflow                   2.6.2

tensorflow-estimator         2.6.0

tensorflow-gpu               2.8.0

tensorflow-io-gcs-filesystem 0.24.0

termcolor                    1.1.0

tf-estimator-nightly         2.8.0.dev2021122109

urllib3                      1.22

wheel                        0.37.1

zipp                         3.6.0
