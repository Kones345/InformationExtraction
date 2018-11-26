import nltk
from nltk import ne_chunk, pos_tag, word_tokenize, sent_tokenize
from regex_store import *
import re
from word2number import w2n
from Tagger import Tagger
from nltk.tag import DefaultTagger
from regex_store import *
import universities
from nltk.tree import Tree
from nltk.corpus import stopwords
import nltk.data
from dateutil import parser as time_parser

#Class to extract data from the text
class DataExtractor():

    #Instance of the tagger class
    tagger = Tagger()

    #Regex to be used lated in the known loacation
    knownLocationRegx = re.compile(knownLocationRegxStr)
    
    #KNOWN SPEAKERS
    knownSpeakers = set()

    #KNOWN LOCATIONS
    knownLocations = set()

    def __init__(self):
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    def extractTime(self,text):

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
    
    def cleanLoc(self,x):
        try:
            result = w2n.word_to_num(x)
        except:
            result = None
        return  result is None and x != "" and (x.isdigit()) == False and x != "room" and x!= "Room"
    
    def extractLocation(self, header, body):
        location_regx = re.compile(location_regx_str, re.IGNORECASE)
        locations = set()
        locationList = []
        locations = {match.group(1) for match in location_regx.finditer(header)}
        locations |= {match.group(1) for match in location_regx.finditer(body)}

        if(len(locations) == 0):
            joined = ' '.join(body.split())
            locationList = self.tagger.nerStanford(joined, "LOCATION")
            for x in locationList:
                if x in self.knownLocations:
                    locations.add(x)

        if(len(locations) == 0):
            return set(locationList)
        else:
            return locations


    def get_continuous_chunks(self, chunked):
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

    def cleanSpeakerList(self, speakers):
        final = set()
        temp = set()
        for x in speakers:
            x = re.sub(",", "", x)
            x = re.sub("(\s?)(?:,|-|\/)(\s?).*", "", x)
            x = re.sub("(\s?)\(.*", "", x)
            temp.add(x)
        
        for x in temp:
            if x != '':
                self.knownSpeakers.add(x)
                final.add(x)

        # print("CLEAN SPEAKERS; ", speakers)
        return final
    def extractSpeaker(self, header, body):
        speakerList = []
        speaker_regex = re.compile(speaker_regx_str, re.IGNORECASE)
        flatten = lambda l: [item for sublist in l for item in sublist]
        speakerList.append(speaker_regex.findall(header))
        speakerList = flatten(speakerList)
        if(len(speakerList) > 0):
            return self.cleanSpeakerList(speakerList)
        speakerList.append(speaker_regex.findall(body))
        speakerList = flatten(speakerList)
        global knownSpeakers
        if(len(speakerList) == 0):
            for s in self.knownSpeakers:
                    match = re.search(s, body.lower())
                    if match is not None:
                        self.knownSpeakers.add(s)
                        speakerList.append(s)

        if(len(speakerList) == 0): 
            joined = ' '.join(body.split())
            speakerList = self.tagger.nerStanford(joined, "PERSON")

        return self.cleanSpeakerList(speakerList)
