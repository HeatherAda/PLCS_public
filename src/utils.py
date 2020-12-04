import os
import numpy as np

def readTransferLabel(transferLabelFile):
	f = open(transferLabelFile)

	auditorLabelList = []
	transferLabelList = []
	trueLabelList = []

	for rawLine in f:
		
		if "transfer" in rawLine:
			continue
		
		line = rawLine.strip().split("\t")
		lineLen = len(line)

		auditorLabelList.append(float(line[0]))
		transferLabelList.append(float(line[1]))
		trueLabelList.append(float(line[2]))

	f.close()

	return auditorLabelList, transferLabelList, trueLabelList

def readFeatureLabelCSV(csvFile):
    f = open(csvFile)

    firstLine = False

    featureMatrix = []
    label = []

    firstLine = f.readline()
    
    posFeatureMatrix = []
    posLabel = []
    negFeatureMatrix = []
    negLabel = []

    for rawLine in f:
        line = rawLine.strip().split(",")
        lineLen = len(line)

        featureList = []
        for lineIndex in range(lineLen-1):
            featureVal = float(line[lineIndex])
            featureList.append(featureVal)

#         featureMatrix.append(featureList)
        if line[lineLen-1] == "FALSE":
            negFeatureMatrix.append(featureList)
            negLabel.append(0.0)
        else:
            posFeatureMatrix.append(featureList)
            # print(line[lineLen-1])
            posLabel.append(1.0)
    
    negFeatureMatrix = random.sample(negFeatureMatrix, len(posLabel))
    negLabel = random.sample(negLabel, len(posLabel))
    
    featureMatrix = np.vstack((negFeatureMatrix, posFeatureMatrix))
    label = np.hstack((negLabel, posLabel))
    
    f.close()

    return featureMatrix, label

def readFeatureLabel(featureLabelFile):
	f = open(featureLabelFile)

	featureMatrix = []
	labelList = []

	for rawLine in f:
		line = rawLine.strip().split("\t")

		lineLen = len(line)

		featureSample = []
		for lineIndex in range(lineLen-1):
			featureVal = float(line[lineIndex])
			featureSample.append(featureVal)

		labelList.append(float(line[lineLen-1]))

		featureMatrix.append(featureSample)

	f.close()

	return featureMatrix, labelList

def readSensorData():
	raw_pt = [i.strip().split('\\')[-1][:-5] for i in open('../../dataset/sensorType/sdh_soda_rice/rice_names').readlines()]
	tmp = np.genfromtxt('../../dataset/sensorType/rice_hour_sdh', delimiter=',')
	label = tmp[:,-1]

	fn = get_name_features(raw_pt)

	featureMatrix = fn
	labelList = label

	return featureMatrix, labelList

def readFeatureFile(featureFile, labelIndex):
	f = open(featureFile)

	featureMatrix = []
	labelList = []

	for rawLine in f:
		line = rawLine.strip().split("\t")

		lineLen = len(line)

		featureSample = []
		for lineIndex in range(lineLen):
			featureVal = float(line[lineIndex])
			featureSample.append(featureVal)
		
		labelList.append(labelIndex)

		featureMatrix.append(featureSample)

	f.close()

	return featureMatrix, labelList

def writeFile(fileSrc, modelVersion, totalAccList, fileType):
	totalACCFile = modelVersion+"_"+fileType+".txt"
	totalACCFile = os.path.join(fileSrc, totalACCFile)

	f = open(totalACCFile, "w")
	for i in range(10):
		totalAlNum = len(totalAccList[i])
		for j in range(totalAlNum):
			f.write(str(totalAccList[i][j])+"\t")
		f.write("\n")
	f.close()