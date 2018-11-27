import time
start_time = time.time()
#Set up corpus
import nltk
from os import listdir
from os.path import isfile,join
import os
import re
import nltk.data
import sys
from pathlib import Path
from Utils import Utils

from regex_store import *
from DataExtractor import DataExtractor
from Tagger import Tagger
from time import sleep
import os
import os.path
import errno

#Setting up directory
mypath = os.getcwd() + '/data/untagged/'
directory = os.fsencode(mypath)

totalFiles = sum(len(files) for _, _, files in os.walk(mypath))

print("\nTagging progress beginning. Get a brew, it'll take a while... \n\n")

extractor = DataExtractor()
extractor.train('data/training')
tagger = Tagger()
tagger.tagSeminar(mypath,directory, extractor, totalFiles)
seconds = time.time() - start_time
m, s = divmod(seconds, 60)
print("There program has been running for {0} minutes and {1} seconds".format(round(m),round(s)))
