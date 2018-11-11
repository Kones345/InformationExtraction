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

#Train corpus 
# train_sents = brown.tagged_sents()[:48000]
# test_sents = brown.tagged_sents()[48000:]


knownLocationRegxStr = '<location>(.+)<\/location>'
knownLocationRegx = re.compile(knownLocationRegxStr)
deadTag = '<\/sentence>'
re.compile(deadTag)
deadTag1 = '<\/paragraph>'
re.compile(deadTag1)
#KNOWN LOCATIONS
knownLocations = set()

knownSpeakersRegxStr = '<speaker>(?:Dr|Mr|Ms|Mrs|Prof|Sir|Professor)?\.?\s?([a-zA-Z ]+),?\s?(?:PhD)?<\/speaker>'
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
                knownSpeakers.add(speaker)
        
        locations = set(re.findall(knownLocationRegx, text))
        if len(locations) > 0:
            for loc in locations:
                loc = re.sub(deadTag, "", loc)
                loc = re.sub(deadTag, "", loc)
                knownLocations.add(loc)


print(knownLocations)
print(knownSpeakers)



#Setting up directory
mypath = os.getcwd() + '/untagged/'


onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
directory = os.fsencode(mypath)

def backoff_tagger(train_sents, tagger_classes, backoff=None):

    for cls in tagger_classes :
        backoff = cls(train_sents, backoff=backoff)
    return backoff

# tagger = backoff_tagger(train_sents, [UnigramTagger, BigramTagger, TrigramTagger], backoff=DefaultTagger('NN'))

def findAndReplaceTime(text):

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
        start = '<stime> ' + result2.group(1) + ' </stime>'
        end = '<etime> ' + result2.group(3) + ' </etime>'
        text = re.sub(str(result2.group(1)), start, text)
        text = re.sub(str(result2.group(3)), end, text)
    elif(result5 is not None):
        start = '<stime> ' + result5.group(1) + ' </stime>'
        end = '<etime> ' + result5.group(2) + ' </etime>'
        text = re.sub(str(result5.group(1)), start, text)
        text = re.sub(str(result5.group(2)), end, text)
    elif(result6 is not None):
        start = '<stime> ' + result6.group(1) + ' </stime>'
        end = '<etime> ' + result6.group(3) + ' </etime>'
        text = re.sub(str(result6.group(1)), start, text)
        text = re.sub(str(result6.group(3)), end, text)  
    elif(result4 is not None):
        start = '<stime> ' + result4.group(1) + ' </stime>'
        text = re.sub(str(result4.group(1)), start, text)
    elif(result1 is not None):
        start = '<stime> ' + result1.group(1) + ' </stime>'
        end = '<etime> ' + result1.group(2) + ' </etime>'
        text = re.sub(str(result1.group(1)), start, text)
        text = re.sub(str(result1.group(2)), end, text)
    elif(result3 is not None):
        start = '<stime> ' + result3.group(0) + ' </stime>'
        text = re.sub(str(result3.group(0)), start, text)
    return text

def get_continuous_chunks(chunked):
    # chunked = ne_chunk(pos_tag(word_tokenize(text)))
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

#corpus = ""

#Setting up data holders
emailHeaders = []
emailBodies = {}
counter = 0

#TIME PATTERNS
timePattern1 = '(\d{1,2}:\d{2}) - (\d{1,2}:\d{2})' # dd:dd - dd:dd
am_pm = '(AM|PM|am|pm|A\.M\.|P\.M\.|a\.m\.|p\.m\.| AM| PM| am| pm| A\.M\.| P\.M\.| a\.m\.| p\.m\.)'
timePattern2 = "(\d{1,2}:\d{2}" + am_pm + ') - (\d{1,2}:\d{2}' + am_pm + ')'#dd:dd am - dd:dd am
timePattern3 =  '\d{1,2}:\d{2}' #dd:dd || d:dd
timePattern4 = '(' + timePattern3 + am_pm + ')'#dd:dd AM
timePattern5 = '(' + timePattern3 + ')' + ' - ' + timePattern4
timePattern6 = timePattern4 + ' - (' + timePattern3 + ')'  

header_body_regx_str = r'([\s\S]+(?:\b.+\b:.+\n\n|\bAbstract\b:))([\s\S]*)'

#Extracting files
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".txt"): 
        with open(mypath + filename, 'r', encoding='utf-8') as f:
            #Read in email
            placeholder= f.read()
            #Splits the email on the word abstract
            # portions = placeholder.split(header_body_regx_str)

            # print(len(portions))
            try:
                header, body = re.search(header_body_regx_str, placeholder).groups()
            except:
                print(filename)
                continue
            
            print("HEADER:")
            print()
            print(header)
            print()
            print("BODY:")
            print()
            print(body)
            print()

            #Empty File
            # if(len(portions) < 2):
            #     print(filename)
            #     print(portions)
            #     continue
            # #More than one abstract found
            # elif (len(portions) > 2):
            #     rest = portions[1:]
            #     portions[1] = "Abstract: ".join(rest)

            #Appends the body to a dictionary relatung to the position of the header in the header list

            # splittter = ' '.join(portions[1].split())
            # clean = re.sub("[^a-zA-Z\d\s:]{2,}", "", splittter)
            # clean = re.sub("- - - -", "", clean)
            # # location_regx_str = r'(?:\b(?:Place|Location|Where)\b:\s*)(.*)'
            # # location_reg = re.compile(location_regx_str, re.IGNORECASE)
            # # locations = {match.group(1) for match in location_reg.finditer(portions[0])}
            # # locations |= {match.group(1) for match in location_reg.finditer(clean)}
            # # print(locations)
            # sents = nltk.tokenize.sent_tokenize(clean)
            # for i in range(0, len(sents)):
            #     sents[i] = nltk.tokenize.word_tokenize(sents[i])
            
            # tagged = tagger.tag_sents(sents)
            # flatten = lambda l: [item for sublist in l for item in sublist]
            # tagged = flatten(tagged)
            # print(tagged)
            # # chunked = ne_chunk(pos_tag(word_tokenize(clean)))
            # chunked = ne_chunk(tagged)
     
            
            # entities = get_continuous_chunks(chunked)
            # print(entities)

            # emailBodies[counter] = portions[1]
            
            # #Tag Time in Header
            # portions[0] = findAndReplaceTime(portions[0])
            # emailHeaders.append(portions[0])
            # print(portions[0])

        counter+=1
        continue
# corpus = re.sub("<.*?>", "", corpus)
