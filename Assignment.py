
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
from nltk.corpus import brown

from nltk.tag import UnigramTagger
from nltk.tag import BigramTagger
from nltk.tag import TrigramTagger
from nltk.tag import DefaultTagger
import nltk.data
from nltk.tree import Tree
from nltk import ne_chunk, pos_tag, word_tokenize
import sys
from pathlib import Path

from regex_store import *
import universities

from word2number import w2n


#Train corpus 
train_sents = brown.tagged_sents()[:48000]
test_sents = brown.tagged_sents()[48000:]

def backoff_tagger(train_sents, tagger_classes, backoff=None):

    for cls in tagger_classes :
        backoff = cls(train_sents, backoff=backoff)
    return backoff

tagger = backoff_tagger(train_sents, [UnigramTagger, BigramTagger, TrigramTagger], backoff=DefaultTagger('NN'))

knownLocationRegx = re.compile(knownLocationRegxStr)
re.compile(deadTag)
re.compile(deadTag1)
#KNOWN LOCATIONS
knownLocations = set()

uni = universities.API()
allUnis = uni.get_all()

for x in allUnis:     
    knownLocations.add(x.name)

# length = sum(1 for _ in allUnis)
# print(length)
# knownLocations

knownSpeakersRegx = re.compile(knownSpeakersRegxStr)
#KNOWN SPEAKERS
knownSpeakers = set()

trainingPath = '/Users/Adam/nltk_data/corpora/seminarTraining'
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
        
        locations = set(re.findall(knownLocationRegx, text))
        if len(locations) > 0:
            for loc in locations:
                loc = re.sub(deadTag, "", loc)
                loc = re.sub(deadTag, "", loc)
                loc = re.sub(r'[^\w\s]','',loc)
                knownLocations.add(loc)

##
# FINDING THE FILE INFORMATION
##

def extractTime(text):

    time1 = re.compile(timePattern1)
    time2 = re.compile(timePattern2)
    time3 = re.compile(timePattern3)
    time4 = re.compile(timePattern4)
    time5 = re.compile(timePattern5)
    time6 = re.compile(timePattern6)

    #Searches the header for a string with each pattern
    result1 = re.search(time1, text)
    result2 = re.search(time2, text)
    result3 = re.search(time3,text)
    result4 = re.search(time4, text)
    result5 = re.search(time5, text)
    result6 = re.search(time6, text)

    #Checks the possible matches and subs time into correct place
    if result2 is not None:
        return str(result2.group(1)), str(result2.group(3))
    elif(result5 is not None):
        return str(result5.group(1)), str(result5.group(2))
    elif(result6 is not None):
        return str(result6.group(1)), str(result6.group(3))
    elif(result4 is not None):
        return str(result4.group(1)), None
    elif(result1 is not None):
        return str(result1.group(1)), str(result1.group(2))
    elif(result3 is not None):
        return str(result3.group(0)), None
    return None, None


def cleanLoc(x):
    try:
        result = w2n.word_to_num(x)
    except:
        result = None
    return  result is None and x != "" and (x.isdigit()) == False and x != "room" and x!= "Room"

def extractLocationREGEX(header, body):
    location_regx = re.compile(location_regx_str, re.IGNORECASE)
    locations = {match.group(1) for match in location_regx.finditer(header)}
    locations |= {match.group(1) for match in location_regx.finditer(body)}

    pos_location_regx = re.compile(pos_location_regx_str, re.IGNORECASE)
    tagged_body = tagPOS(body)
    
    taggedLocations = set()
    tags = pos_location_regx.finditer(tagged_body)
    tagged_locations = {match.group(1) for match in pos_location_regx.finditer(tagged_body)}
    tagged_locations = {re.sub(pos_tags_regx_str, '', location).strip() for location in tagged_locations}

    res = locations.union(tagged_locations)
    if (len(res) == 0):
        for loc in knownLocations:
            search = re.search(str(loc), body)
            if (search is not None):
                locations.add(loc)

    res = set(filter(cleanLoc, res))
    knownLocations = knownLocations.union(res)
    # res = set(filter(cleanLoc, res))

    return res

def get_continuous_chunks(chunked):
    prev = None
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue
    return continuous_chunk

def extractLocationNER(header, body):
    location_regx = re.compile(location_regx_str, re.IGNORECASE)
    locations = {match.group(1) for match in location_regx.finditer(header)}
    locations |= {match.group(1) for match in location_regx.finditer(body)}

    splittter = ' '.join(body)
    clean = re.sub("[^a-zA-Z\d\s:]{2,}", "", splittter)
    clean = re.sub("- - - -", "", clean)
    chunked = ne_chunk(pos_tag(word_tokenize(clean)))
    # chunked = ne_chunk(tagged)
    entities = get_continuous_chunks(chunked)
    return entities

def tagPOS(body):
    sents = nltk.tokenize.sent_tokenize(body)
    for i in range(0, len(sents)):
        sents[i] = nltk.tokenize.word_tokenize(sents[i])
    tagged = tagger.tag_sents(sents)
    flatten = lambda l: [item for sublist in l for item in sublist]
    tagged = flatten(tagged)
    processed_body = ['{}{{*{}*}}'.format(word, tag) for (word, tag) in tagged]

    return ' '.join(processed_body)


#Setting up directory
mypath = os.getcwd() + '/untagged/'
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


            stime, etime = extractTime(header)
            # tagged = tagPOS(body)
            # print(tagged)
            locations = extractLocationREGEX(header, body)
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


