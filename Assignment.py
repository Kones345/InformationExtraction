
#My current approach is to use tagged emails as training to extract known locations and speakers from them
#After this I will extract the header and body
#Extract the time from the header
#Extract the location from the body
#Add found locations to the set of known locations
#Repeat this for speakkers
#Once I have location and speakers I will tag sentences and paragraphs with regex


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


re.compile(deadTag)
re.compile(deadTag1)

extractor = DataExtractor()

knownSpeakersRegx = re.compile(knownSpeakersRegxStr)
#KNOWN SPEAKERS
knownSpeakers = set()

trainingPath = 'data/training'
pathlist = Path(trainingPath).glob('**/*.txt')
for path in pathlist:
    p = str(path)
    with open(p, 'r', encoding='utf-8') as f:
        text = (f.read()).lower()
        speakers = set(re.findall(knownSpeakersRegx, text))
        if len(speakers) > 0:
            for speaker in speakers:
                speaker = re.sub(r'[^\w\s]','',speaker)
                knownSpeakers.add(speaker)
        
        locations = set(re.findall(extractor.knownLocationRegx, text))
        if len(locations) > 0:
            for loc in locations:
                loc = re.sub(deadTag, "", loc)
                loc = re.sub(deadTag, "", loc)
                loc = re.sub(r'[^\w\s]','',loc)
                extractor.knownLocations.add(loc)




#Setting up directory
mypath = os.getcwd() + '/data/untagged/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
directory = os.fsencode(mypath)

header_body_regx_str = r'([\s\S]+(?:\b.+\b:.+\n\n|\bAbstract\b:))([\s\S]*)'
locationFound = 0
noLocation = 0
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
            # tagged = tagPOS(body)
            # print(tagged)
            locations = extractor.extractLocationREGEX(header, body)
            if(len(locations) > 0):
                locationFound +=1
            else:
                noLocation += 1
            # print("TIME: " + stime + " ", etime)
            print()
            print(filename)
            print()
            # print("HEADER:")
            # print()
            # print(header)
            # print()
            # print("BODY:")
            # print()
            # print(body)
            # print()
            print("LOCATIONS REGEX: ")
            print(locations)
            print()
            # print("LOCATIONS NER: ")
            # print()
            # locations1 = extractLocationNER(header, body)
            # print(locations1)
            # print(knownLocations)
        
        continue

print(locationFound, noLocation)
print(locationFound/(locationFound + noLocation))


