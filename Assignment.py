import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import time
from DataExtractor import DataExtractor
from Tagger import Tagger
from Ontology import Ontology
import os
from Evaluation import Evaluation


# Clock to time what the running of the program
start_time = time.time()

# Setting up directory
mypath = os.getcwd() + '/data/seminar_testdata/test_untagged/'
trainingPath = 'data/training'
directory = os.fsencode(mypath)

# Runs the ontology classification
print('Running Ontology Classification: \n')
ontology = Ontology()
ontology.run(mypath)

# Begins tagging
print("\nTagging progress beginning. Get a brew, it'll take a while... \n")
extractor = DataExtractor()

# Trains our model
extractor.train(trainingPath)
tagger = Tagger()

# Tags all emails in the directory given
tagger.tag_seminar(mypath, directory, extractor)

# Calculates how long the program took
seconds = time.time() - start_time
m, s = divmod(seconds, 60)
print("The program has been running for {0} minutes and {1} seconds \n".format(round(m), round(s)))

# Evaluates results
eval = Evaluation()
eval.run()

