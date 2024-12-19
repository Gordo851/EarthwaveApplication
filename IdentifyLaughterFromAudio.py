# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 16:17:16 2024

@author: gordo
"""
import json 
from keras.models import model_from_json
import pandas as pd
from pydub import AudioSegment
import wave
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
bundle = torchaudio.pipelines.HUBERT_BASE
model1 = bundle.get_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
This .py allows the automatic analysis of short .wav audio files.

The code will output a graph showing the estimated probability of laughter in
the time dimension for the clip as well as the estimated time stamp of where
laughter occurred. 

Longer audio files will take up more ram, currently only the first 10 seconds
of an audio file is tested - to lengthen this increase the max_length variable.
"""



audioFileName = "record_out.wav"
max_length = 10
minGap = 2
minimumProbabilityForLaugh = 0.1

filenames = {
	"rep0": ['ModelFromPaper/2layers100nodesper/kFoldResults/6FOLDlaughrep0NoUnderSample.json', "ModelFromPaper/2layers100nodesper/h5ModelSaveSpace"],
	}



# This extracts numerical features from the wav file using a publically available
# transformer model
def extractFeatures(audioFileName):
	frameFeatures = []
	newAudio = AudioSegment.from_wav(audioFileName)
	f1 = wave.open(audioFileName, "r")
	num_channels_file1 = int(f1.getnchannels())			
	
	if num_channels_file1 > 1:
		newAudio.set_channels(1)
	duration = newAudio.duration_seconds
	if duration > max_length:
		end = max_length*1000
		newAudio = newAudio[0:end]
		duration = newAudio.duration_seconds
		newAudio.export("Sample.wav", format="wav")
		waveform, sample_rate = torchaudio.load("Sample.wav")
	else:
		waveform, sample_rate = torchaudio.load(audioFileName)
	
	waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
	
	with torch.inference_mode():
		features, _ = model1.extract_features(waveform)
	
	for y in range(0,len(features[0][0])):		
		resultsHolder = []
		for x in range(0,12):
			resultsHolder.append((features[x][0][y]).numpy())
		avg = (sum(resultsHolder))/len(resultsHolder)
		frameFeatures.append(avg)
	df = pd.DataFrame(frameFeatures)
	return df

# This uses a pretrained model to generate the probability of a frame being laughter
def createLaughterPredictions(features):
	key = "rep0"
	with open(filenames[key][0]) as json_file:
	    data = json.load(json_file)
	x=0
	keysOfModel = list(data.keys())
	modelKey = keysOfModel[x]
	modelData = data[modelKey][1]
	h5Name = filenames[key][1] + "/" + modelKey +  key +"NoUndersampledmodel.h5"
	loaded_model = model_from_json(modelData)
	loaded_model.load_weights(h5Name)
	y_pred = loaded_model.predict(features, verbose = 1) 
	features["predictions"] = y_pred
	return features

# Small graph function to display results
def showGraphOfPredictions(df):	
	plt.plot(df["predictions"], linestyle = 'dotted', label="Laughter Prediction")
	plt.axhline(y = minimumProbabilityForLaugh, color = 'r', linestyle = '-', label="Cut-off For Laughter") 
	totalSeconds = len(df["predictions"])*0.02
	labels=np.arange(0, totalSeconds+1, step=5)
	ticks = labels*50
	plt.xticks(ticks=ticks, labels=labels)
	ax = plt.gca()
	ax.set_ylim([0, 1])
	plt.title("Laughter Prediction")
	plt.xlabel("Time (s)")
	plt.ylabel("Estimated Probability of Laughter")

	plt.show()


# Merges timestamps that are close together to tidy up the presented results
def mergeUnderMinimum(timeStamps, minGap):
	minGapInFrames = minGap / 0.02
	if len(timeStamps) == 0:
		return []
	merged = [timeStamps.pop(0)]
	last = merged[0]

	while timeStamps:
		current = timeStamps.pop(0)
		if len(timeStamps) >= 1:
			Next = timeStamps[0]
			if current - last > minGapInFrames and Next-current > minGapInFrames:
				merged.append(current)
				last = current
			else:
				ind=1
				while True:
					if len(timeStamps) >= 1:
						Next = timeStamps.pop(0)
						if Next-current > minGapInFrames:
							halfDistance = int((Next-current)/2)
							midPoint = current + halfDistance
							merged.append(midPoint)
							last = current
							break
						ind = ind + 1
					else:
						break
				
	return merged



# Outputs merged laughter timestamps to console
def displayLaughterTimeStamps(df):
	aboveMinimum = list(df.loc[df.predictions.ge(minimumProbabilityForLaugh)].index)
	aboveMinimumMerged = mergeUnderMinimum(aboveMinimum, minGap)
	if len(aboveMinimumMerged) == 0:
		print("No laughter detected. Have you tried telling funnier jokes?")
		return
	print("Laughter was detected at the following timestamps:")
	for frameStamp in aboveMinimumMerged:
		actualTime = frameStamp * 0.02
		s = "{time:.2f} seconds"
		print(s.format(time=actualTime))


def main():
	features = extractFeatures(audioFileName)
	df = createLaughterPredictions(features)
	displayLaughterTimeStamps(df)
	showGraphOfPredictions(df)


main()
