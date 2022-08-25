from config import car_config as config
import mxnet as mx
import argparse
import logging 
import os

ap = argparse.ArgumentParser()
ap.add_argument("-v","--vgg",required=True,help="path to pre trained vggnet")
ap.add_argument("-c","--checkpoints",required=True,help="path to checkpoints directory")
ap.add_argument("-p","--prefix",required=True,help="name of model prefix")
ap.add_argument("-s","--startepoch",type=int,default=0,help="epoch to start from")
args = vars(ap.parse_args())

logging.basicConfig(level=logging.DEBUG,filename="training_{}.log".format(args["startepoch"]),filemode="w")

batchsize = config.BATCH_SIZE
trainIter = mx.io.ImageRecordIter(path_imgrec = config.TRAIN_MX_REC,data_shape = (3,224,224),batch_size = batchsize, rand_crop = True,
    rand_mirror = True,rotate = 15, max_shear_ratio = 0.1,mean_r = config.R_MEAN,mean_b = config.B_MEAN ,mean_g = config.G_MEAN,preprocess_threads=2)

valIter = mx.io.ImageRecordIter(path_imgrec = config.VAL_MX_REC,data_shape = (3,224,224),batch_size = batchsize,
    mean_r = config.R_MEAN,mean_b = config.B_MEAN ,mean_g = config.G_MEAN)

opt = mx.optimizer.SGD(learning_rate = 1e-3,momentum = 0.9,wd=0.0005,rescale_grad = 1.0/batchsize)
checkpoints_path = os.path.sep.join([args["checkpoints"],args["prefix"]])
argParams = None
auxParams = None
allowMissing = False

if args["startepoch"] <=0:
    #load the model
    print("[INFO] loading pre-trained model...")
    (symbol,argParams,auxParams) = mx.model.load_checkpoint(args["vgg"],0)
    allowMissing = True

    #grab the layers from the pre-trained model, then find dropout layer prior toy the final fc layer
    layers = symbol.get_internals()
    net = layers["drop7_output"]
    net = mx.sym.FullyConnected(data=net,num_hidden=config.NUM_CLASSES,name="fc8")
    net = mx.sym.SoftmaxOutput(data=net,name="softmax")

    argParams = dict({K:argParams[K] for K in argParams if "fc8" not in K})

else:
    print("[INFO] loading epoch {}...".format(args["startepoch"]))
    (net,argParams,auxParams) = mx.model.load_checkpoint(checkpoints_path,args["startepoch"])

batchEndCBs = [mx.callback.Speedometer(batchsize,50)]
epochEndCBs = [mx.callback.do_checkpoint(checkpoints_path)]
metries =[mx.metric.Accuracy(),mx.metric.TopKAccuracy(top_k=5),mx.metric.CrossEntropy()]
print("[INFO] training network...")

model = mx.mod.Module(symbol=net,context = mx.gpu(0))
model.fit(trainIter,eval_data=valIter,num_epoch=65,begin_epoch=args["startepoch"],initializer=mx.initializer.Xavier(),
    arg_params=argParams,aux_params=auxParams,optimizer=opt,allow_missing=allowMissing,eval_metric=metries,
    batch_end_callback=batchEndCBs,epoch_end_callback=epochEndCBs)

