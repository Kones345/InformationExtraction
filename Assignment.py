import time
from DataExtractor import DataExtractor
from Tagger import Tagger
import os

start_time = time.time()

# For file reading

# Classes to extract and tag seminars


# Setting up directory
mypath = os.getcwd() + '/data/untagged/'
trainingPath = 'data/training'
directory = os.fsencode(mypath)

# Counts the number of files in the working directory
totalFiles = sum(len(files) for _, _, files in os.walk(mypath))

print("\nTagging progress beginning. Get a brew, it'll take a while... \n\n")

extractor = DataExtractor()

# Trains our model
extractor.train(trainingPath)
tagger = Tagger()

# Tags all emails in the directory given
tagger.tagSeminar(mypath, directory, extractor, totalFiles)

# Calculates how long the program took
seconds = time.time() - start_time
m, s = divmod(seconds, 60)
print("There program has been running for {0} minutes and {1} seconds".format(round(m), round(s)))
