import nltk
import re
from word2number import w2n
from regex_store import *
from nltk.tree import Tree
import nltk.data
from pathlib import Path


# Class to extract data from the text
class DataExtractor:
    # Regex to be used later in the known location
    knownLocationRegx = re.compile(knownLocationRegxStr)

    # KNOWN SPEAKERS
    knownSpeakers = set()

    # KNOWN LOCATIONS
    knownLocations = set()

    def __init__(self):
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    def train(self, directory):
        knownSpeakersRegx = re.compile(knownSpeakersRegxStr)

        re.compile(deadTag)
        re.compile(deadTag1)
        pathlist = Path(directory).glob('**/*.txt')
        for path in pathlist:
            p = str(path)
            with open(p, 'r', encoding='utf-8') as f:
                text = (f.read()).lower()
                speakers = set(re.findall(knownSpeakersRegx, text))
                if len(speakers) > 0:
                    for speaker in speakers:
                        speaker = re.sub(r'[^\w\s]', '', speaker)
                        self.knownSpeakers.add(speaker)

                locations = set(re.findall(self.knownLocationRegx, text))
                if len(locations) > 0:
                    for loc in locations:
                        loc = re.sub(deadTag, "", loc)
                        loc = re.sub(deadTag1, "", loc)
                        loc = re.sub(r'[^\w\s]', '', loc)
                        self.knownLocations.add(loc)
        print('Training: ✅')

    @staticmethod
    def extractTime(text):

        time1 = re.compile(timePattern1)
        time2 = re.compile(timePattern2)
        time3 = re.compile(timePattern3)
        time4 = re.compile(timePattern4)
        time5 = re.compile(timePattern5)
        time6 = re.compile(timePattern6)

        # Searches the header for a string with each pattern
        result1 = re.search(time1, text)
        result2 = re.search(time2, text)
        result3 = re.search(time3, text)
        result4 = re.search(time4, text)
        result5 = re.search(time5, text)
        result6 = re.search(time6, text)

        # Checks the possible matches and subs time into correct place
        if result2 is not None:
            return str(result2.group(1)), str(result2.group(3))
        elif result5 is not None:
            return str(result5.group(1)), str(result5.group(2))
        elif result6 is not None:
            return str(result6.group(1)), str(result6.group(3))
        elif result4 is not None:
            return str(result4.group(1)), None
        elif result1 is not None:
            return str(result1.group(1)), str(result1.group(2))
        elif result3 is not None:
            return str(result3.group(0)), None
        return None, None

    def cleanLoc(self, x):
        try:
            result = w2n.word_to_num(x)
        except:
            result = None
        return result is None and x != "" and (x.isdigit()) == False and x != "room" and x != "Room"

    def extractLocation(self, header, body, tagger):
        location_regx = re.compile(location_regx_str, re.IGNORECASE)
        locations = set()
        locationList = []
        locations = {match.group(1) for match in location_regx.finditer(header)}
        locations |= {match.group(1) for match in location_regx.finditer(body)}

        if len(locations) == 0:
            joined = ' '.join(body.split())
            locationList = tagger.nerStanford(joined, "LOCATION")
            for x in locationList:
                if x in self.knownLocations:
                    locations.add(x)

        if len(locations) == 0:
            for loc in self.knownLocations:
                if loc in body:
                    locations.add(loc)

        if len(locations) == 0:
            self.knownLocations = self.knownLocations.union(locationList)
            return set(locationList)
        else:
            return locations

    @staticmethod
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

        return final

    def extractSpeaker(self, header, body, tagger):
        speakerList = []
        speaker_regex = re.compile(speaker_regx_str, re.IGNORECASE)
        flatten = lambda l: [item for sublist in l for item in sublist]
        speakerList.append(speaker_regex.findall(header))
        speakerList = flatten(speakerList)
        if len(speakerList) > 0:
            return self.cleanSpeakerList(speakerList)
        speakerList.append(speaker_regex.findall(body))
        speakerList = flatten(speakerList)

        if len(speakerList) == 0:
            for s in self.knownSpeakers:
                match = re.search(s, body.lower())
                if match is not None:
                    self.knownSpeakers.add(s)
                    speakerList.append(s)

        if len(speakerList) == 0:
            joined = ' '.join(body.split())
            speakerList = tagger.nerStanford(joined, "PERSON")

        self.knownSpeakers = self.knownSpeakers.union(self.cleanSpeakerList(speakerList))

        return self.cleanSpeakerList(speakerList)
