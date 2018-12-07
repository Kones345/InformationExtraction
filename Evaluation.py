import re
from regex_store import *


class Evaluation:

    def extractTestData(self, filename):
        test_file_path = '/Users/Adam/Documents/BRUM/SecondYear/Modules/NLP/Assignment/data/seminar_testdata/test_tagged/'
        file = open(test_file_path + filename, 'r')
        contents = file.read()
        speaker_regex = re.compile(knownSpeakersRegxStr)
        speakers = set(speaker_regex.findall(contents))
        location_regex = re.compile(knownLocationRegxStr)
        locations = set(re.findall(location_regex, contents))
        print(speakers)
        print(locations)


if __name__ == '__main__':
    eval = Evaluation()
    eval.extractTestData('301.txt')
