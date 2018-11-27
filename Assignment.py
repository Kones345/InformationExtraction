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

from regex_store import *
from DataExtractor import DataExtractor

import os
import errno

re.compile(deadTag)
re.compile(deadTag1)

extractor = DataExtractor()
extractor.train('data/training')

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

#Setting up directory
mypath = os.getcwd() + '/data/untagged/'
directory = os.fsencode(mypath)

header_body_regx_str = r'([\s\S]+(?:\b.+\b:.+\n\n|\bAbstract\b:))([\s\S]*)'

#Extracting files
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".txt"): 
        with open(mypath + filename, 'r', encoding='utf-8') as f:
            #Read in email
            placeholder= f.read()

            #Splits the text into header and body
            try:
                header, body = re.search(header_body_regx_str, placeholder).groups()
            except:
                print(filename)
                continue

            
            stime, etime = extractor.extractTime(header)
            locations = extractor.extractLocation(header, body)
            print()
            print(filename)
            print()
       
            speakers = extractor.extractSpeaker(header, body)

            body = extractor.tagger.tag_paragraphs(body)
            body = extractor.tagger.tag_sentences(body)

            seminar = header + '\n\n' + body
            seminar = extractor.tagger.tagTimes(stime, etime,seminar)
            seminar = extractor.tagger.tag_speakers(seminar, speakers)
            seminar = extractor.tagger.tag_locations(locations, seminar)

            outLocation = "out/"
            mkdir_p(outLocation)
            out = open(outLocation + filename,"w+")
            out.write(seminar)
            out.close()

        continue
seconds = time.time() - start_time
m, s = divmod(seconds, 60)
print("There program has been running for {0} minutes and {1} seconds".format(m,round(s)))
