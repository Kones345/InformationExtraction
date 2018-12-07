import re
from regex_store import *

class Evaluation:

    def extractTestData(self, filename):
        test_file_path = '/Users/Adam/Documents/BRUM/SecondYear/Modules/NLP/Assignment/data/seminar_testdata/test_tagged/'
        file = open(test_file_path + filename, 'r')
        contents = file.read()
        speaker_regex = re.compile(knownSpeakersRegxStr)
        speakers = speaker_regex.findall(contents)
        location_regex = re.compile(knownLocationRegxStr)
        locations = re.findall(location_regex, contents)
        stimes = re.findall(stime_regex_str, contents)
        etimes = re.findall(etime_regex_str, contents)
        sents = re.findall(sent_regex_str, contents)
        paras = re.findall(para_regex_str, contents)
        print(speakers)
        # print(locations)
        # print(stimes)
        # print(etimes)
        # print(sents)
        # print(paras)

    def extractOutData(self, filename):
        test_file_path = '/Users/Adam/Documents/BRUM/SecondYear/Modules/NLP/Assignment/out/'
        file = open(test_file_path + filename, 'r')
        contents = file.read()
        speaker_regex = re.compile(knownSpeakersRegxStr)
        speakers = speaker_regex.findall(contents)
        location_regex = re.compile(knownLocationRegxStr)
        locations = re.findall(location_regex, contents)
        stimes = re.findall(stime_regex_str, contents)
        etimes = re.findall(etime_regex_str, contents)
        sents = re.findall(sent_regex_str, contents)
        paras = re.findall(para_regex_str, contents)
        print(speakers)
        # print(locations)
        # print(stimes)
        # print(etimes)
        # print(sents)
        # print(paras)

if __name__ == '__main__':
    eval = Evaluation()
    eval.extractTestData('400.txt')
    eval.e
