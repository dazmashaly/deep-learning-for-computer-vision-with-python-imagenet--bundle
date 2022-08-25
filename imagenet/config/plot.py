# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import argparse
import re
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--network", required=True,
	help="name of network")
ap.add_argument("-d", "--dataset", required=True,
	help="name of dataset")
args = vars(ap.parse_args())
logs = [(5, "training_0.log")]

# initialize the list of train rank-1 and rank-5 accuracies, along
# with the training loss
(trainRank1, trainRank5, trainLoss) = ([], [], [])
# initialize the list of validation rank-1 and rank-5 accuracies,
# along with the validation loss
(valRank1, valRank5, valLoss) = ([], [], [])

# loop over the training logs
for (i, (endEpoch, p)) in enumerate(logs):
	# load the contents of the log file, then initialize the batch
	# lists for the training and validation data
	rows = open(p).read().strip()
	(bTrainRank1, bTrainRank5, bTrainLoss) = ([], [], [])
	(bValRank1, bValRank5, bValLoss) = ([], [], [])
	# grab the set of training epochs
	epochs = set(re.findall(r'Epoch\[(\d+)\]', rows))
	epochs = sorted([int(e) for e in epochs])
    # loop over the epochs
	for e in epochs:
    	# find all rank-1 accuracies, rank-5 accuracies, and loss
		# values, then take the final entry in the list for each
		s = r'Epoch\[' + str(e) + '\] Batch.*'
		q = r'accuracy=.*'
		rank1 = re.findall(s, rows)[-1]
		rrank1 = re.findall(q, rank1)[-1][9:17]
		s = r'Epoch\[' + str(e) + '\] Batch.*'
		q = r'top_k_accuracy_5=.*'
		rank5 = re.findall(s, rows)[-1]
		rrank5 = re.findall(q, rank5)[-1][17:25]
		s = r'Epoch\[' + str(e) + '\] Batch.*'
		q = r'cross-entropy=.*'
		loss = re.findall(s, rows)[-1]
		te = re.findall(q, loss)[-1][14:]
		# update the batch training lists
		bTrainRank1.append(float(rrank1))
		bTrainRank5.append(float(rrank5))
		bTrainLoss.append(float(te))
    # extract the validation rank-1 and rank-5 accuracies for each
	# epoch, followed by the loss
	bValRank1 = re.findall(r'Validation-accuracy=(.*)', rows)
	bValRank5 = re.findall(r'Validation-top_k_accuracy_5=(.*)', rows)
	bValLoss = re.findall(r'Validation-cross-entropy=(.*)', rows)
	# convert the validation rank-1, rank-5, and loss lists to floats
	bValRank1 = [float(x) for x in bValRank1]
	bValRank5 = [float(x) for x in bValRank5]
	bValLoss = [float(x) for x in bValLoss]
    # check to see if we are examining a log file other than the
	# first one, and if so, use the number of the final epoch in
	# the log file as our slice index
	if i > 0 and endEpoch is not None:
		trainEnd = endEpoch - logs[i - 1][0]
		valEnd = endEpoch - logs[i - 1][0]
	# otherwise, this is the first epoch so no subtraction needs
	# to be done
	else:
		trainEnd = endEpoch
		valEnd = endEpoch
    
    # update the training lists
	trainRank1.extend(bTrainRank1[0:trainEnd])
	trainRank5.extend(bTrainRank5[0:trainEnd])
	trainLoss.extend(bTrainLoss[0:trainEnd])
	# update the validation lists
	valRank1.extend(bValRank1[0:valEnd])
	valRank5.extend(bValRank5[0:valEnd])
	valLoss.extend(bValLoss[0:valEnd])

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(trainRank1)), trainRank1,
	label="train_rank1")
plt.plot(np.arange(0, len(trainRank5)), trainRank5,
	label="train_rank5")
plt.plot(np.arange(0, len(valRank1)), valRank1,
	label="val_rank1")
plt.plot(np.arange(0, len(valRank5)), valRank5,
	label="val_rank5")
plt.title("{}: rank-1 and rank-5 accuracy on {}".format(
	args["network"], args["dataset"]))
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")

# plot the losses
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(trainLoss)), trainLoss,
	label="train_loss")
plt.plot(np.arange(0, len(valLoss)), valLoss,
	label="val_loss")
plt.title("{}: cross-entropy loss on {}".format(args["network"],
	args["dataset"]))
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.show()