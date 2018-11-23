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
                extractor.knownSpeakers.add(speaker)
        
        locations = set(re.findall(extractor.knownLocationRegx, text))
        if len(locations) > 0:
            for loc in locations:
                loc = re.sub(deadTag, "", loc)
                loc = re.sub(deadTag, "", loc)
                loc = re.sub(r'[^\w\s]','',loc)
                extractor.knownLocations.add(loc)

#Setting up directory
mypath = os.getcwd() + '/data/untagged/'
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
directory = os.fsencode(mypath)

header_body_regx_str = r'([\s\S]+(?:\b.+\b:.+\n\n|\bAbstract\b:))([\s\S]*)'
locationFound = 0
noLocation = 0
speakersFound  = 0
noSpeakers = 0
foundSpeakers = []
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
            # print("LOCATIONS REGEX: ")
            # print(locations) 
            # print()
            print('Speakers')
            speakers = extractor.extractSpeakerREGEX(header, body)
            # foundSpeakers.append(speakers)
            print(speakers)
            if(len(speakers) > 0):
                speakersFound +=1
                for x in speakers:
                    foundSpeakers.append(x)
            else:
                noSpeakers += 1
            print()
            print("SENTENCES: ", extractor.extractSentences(body))
            print()
            print("PARAGRAPHS: ", extractor.extractParagraphs(body))
            # print("LOCATIONS NER: ")
            # print()
            # locations1 = extractLocationNER(header, body)
            # print(locations1)
            # print(knownLocations)
            # print("STANFORD TAGGING: ", extractor.tagger.nerStanford(body))
        
        continue

# print(locationFound, noLocation)
# print(locationFound/(locationFound + noLocation))
print(foundSpeakers)
print(speakersFound, noSpeakers)
print(speakersFound/(speakersFound + noSpeakers))


