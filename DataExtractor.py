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

class DataExtractor():

    tagger = Tagger()
    # backoff = tagger.backoff_tagger(backoff=DefaultTagger('NN'))
    knownLocationRegx = re.compile(knownLocationRegxStr)
    
    #KNOWN SPEAKERS
    knownSpeakers = set()

    #KNOWN LOCATIONS
    knownLocations = set()

    uni = universities.API()
    allUnis = uni.get_all()

    # for x in allUnis:     
    #     knownLocations.add(x.name)

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
    
    def extractLocationREGEX(self, header, body):
        location_regx = re.compile(location_regx_str, re.IGNORECASE)
        locations = {match.group(1) for match in location_regx.finditer(header)}
        locations |= {match.group(1) for match in location_regx.finditer(body)}

        pos_location_regx = re.compile(pos_location_regx_str, re.IGNORECASE)
        tagged_body = self.tagger.tagPOS(body)
        
        taggedLocations = set()
        tags = pos_location_regx.finditer(tagged_body)
        tagged_locations = {match.group(1) for match in pos_location_regx.finditer(tagged_body)}
        tagged_locations = {re.sub(pos_tags_regx_str, '', location).strip() for location in tagged_locations}

        res = locations.union(tagged_locations)
        global knownLocations
        if (len(res) == 0):
            for loc in self.knownLocations:
                search = re.search(str(loc), body)
                if (search is not None):
                    locations.add(loc)

        res = set(filter(self.cleanLoc, res))
        self.knownLocations = self.knownLocations.union(res)

        return res

    # def get_continuous_chunks(self, text, label):
    #     chunked = ne_chunk(pos_tag(word_tokenize(text)))
    #     prev = None
    #     continuous_chunk = []
    #     current_chunk = []

    #     for subtree in chunked:
    #         if type(subtree) == Tree and subtree.label() == label:
    #             current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
    #         elif current_chunk:
    #             named_entity = " ".join(current_chunk)
    #             if named_entity not in continuous_chunk:
    #                 continuous_chunk.append(named_entity)
    #                 current_chunk = []
    #         else:
    #             continue

    #     return continuous_chunk

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

    def extractSpeakerREGEX(self, header, body):
        speakerList = []
        speaker_regex = re.compile(speaker_regx_str, re.IGNORECASE)
        flatten = lambda l: [item for sublist in l for item in sublist]
        speakerList.append(speaker_regex.findall(header))
        speakerList.append(speaker_regex.findall(body))
        speakerList = flatten(speakerList)
        final = set()
        for x in speakerList:
            if x != []:
                x = re.sub(",", "", x)
                x = re.sub("(\s?)(?:,|-|\/)(\s).*", "", x)
                x = re.sub("(\s?)\(.*", "", x)
                final.add(x)
                self.knownSpeakers.add(x)
        if(len(speakerList) == 0):
            # splitText = body.split()
            #stop_words = set(stopwords.words('english')) 
            joined = ' '.join(body.split())
            results = self.tagger.nerStanford(joined, "PERSON")
            # words = nltk.word_tokenize(joined)
            # filtered = [w for w in words if not w in stop_words]
            # chunked = ne_chunk(pos_tag(filtered))
            # potentialSpeakers = self.get_continuous_chunks(chunked)
            # print("INITIAL CHUNKS: ", chunked)
            # print("SPEAKERS FROM CHUNKING ", potentialSpeakers)
            return results
            # chuinked = nltk.ne_chunk(nltk.pos_tag(words)



        return final

    def extractSentences(self, text):
        return self.sent_detector.tokenize(text.strip())

    
    def extractParagraphs(self, text):
        para = re.compile(paragraphRegex)
        return re.findall(para, text)



    # def extractLocationNER(header, body):
        # location_regx = re.compile(location_regx_str, re.IGNORECASE)
        # locations = {match.group(1) for match in location_regx.finditer(header)}
        # locations |= {match.group(1) for match in location_regx.finditer(body)}

        # splittter = ' '.join(body)
        # clean = re.sub("[^a-zA-Z\d\s:]{2,}", "", splittter)
        # clean = re.sub("- - - -", "", clean)
        # chunked = ne_chunk(pos_tag(word_tokenize(clean)))
        # # chunked = ne_chunk(tagged)
        # entities = get_continuous_chunks(chunked)
        # return entities