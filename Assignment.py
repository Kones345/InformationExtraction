import time
from DataExtractor import DataExtractor
from Tagger import Tagger
from Ontology import Ontology
import os

start_time = time.time()

# Setting up directory
mypath = os.getcwd() + '/data/untagged/'
trainingPath = 'data/training'
directory = os.fsencode(mypath)

print('Running Ontology Classification: \n')
ontology = Ontology()
ontology.run(mypath)

print("\nTagging progress beginning. Get a brew, it'll take a while... \n")
extractor = DataExtractor()

# Trains our model
extractor.train(trainingPath)
tagger = Tagger()

# Tags all emails in the directory given
tagger.tagSeminar(mypath, directory, extractor)

# Calculates how long the program took
seconds = time.time() - start_time
m, s = divmod(seconds, 60)
print("There program has been running for {0} minutes and {1} seconds".format(round(m), round(s)))
