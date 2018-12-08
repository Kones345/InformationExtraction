import re
from regex_store import *
import os

class Evaluation:

    loc_tp = 0
    loc_classified = 0
    loc_true_count = 0

    speaker_tp = 0
    speaker_classified = 0
    speaker_true_count = 0

    stime_tp = 0
    stime_classified = 0
    stime_true_count = 0

    etime_tp = 0
    etime_classified = 0
    etime_true_count = 0

    paragraph_tp = 0
    paragraph_classified = 0
    paragraph_true_count = 0

    sentence_tp = 0
    sentence_classified = 0
    sentence_true_count = 0


    def extractTestData(self, filename):
        test_file_path = '/Users/Adam/Documents/BRUM/SecondYear/Modules/NLP/Assignment/data/seminar_testdata/test_tagged/'
        file = open(test_file_path + filename, 'r')
        contents = file.read()
        speaker_regex = re.compile(speaker_regex_str)
        speakers = speaker_regex.findall(contents)
        location_regex = re.compile(knownLocationRegxStr)
        locations = re.findall(location_regex, contents)
        stimes = re.findall(stime_regex_str, contents)
        etimes = re.findall(etime_regex_str, contents)
        sents = re.findall(sent_regex_str, contents)
        paras = re.findall(para_regex_str, contents)
        # print(speakers)
        # print(locations)
        # print(stimes)
        # print(etimes)
        # print(sents)
        # print(paras)
        return speakers, locations, stimes, etimes, sents, paras

    def extractOutData(self, filename):
        test_file_path = '/Users/Adam/Documents/BRUM/SecondYear/Modules/NLP/Assignment/out/'
        file = open(test_file_path + filename, 'r')
        contents = file.read()
        speaker_regex = re.compile(speaker_regex_str)
        speakers = speaker_regex.findall(contents)
        location_regex = re.compile(knownLocationRegxStr)
        locations = re.findall(location_regex, contents)
        stimes = re.findall(stime_regex_str, contents)
        etimes = re.findall(etime_regex_str, contents)
        sents = re.findall(sent_regex_str, contents)
        paras = re.findall(para_regex_str, contents)
        # print(speakers)
        # print(locations)
        # print(stimes)
        # print(etimes)
        # print(sents)
        # print(paras)
        return speakers, locations, stimes, etimes, sents, paras

    def process_loc_tps(self, actual, output):

        self.loc_classified += len(output)
        self.loc_true_count += len(actual)

        for i in range(0, len(output)):
            try:
                if output[i] == actual[i]:
                    self.loc_tp +=1

            except:
                continue

    def process_speaker_tps(self, actual, output):
        self.speaker_classified += len(output)
        self.speaker_true_count += len(actual)

        for i in range(0, len(output)):
            try:
                if output[i] == actual[i]:
                    self.speaker_tp += 1

            except:
                continue

    def process_stime_tps(self, actual, output):
        self.stime_classified += len(output)
        self.stime_true_count += len(actual)

        for i in range(0, len(output)):
            try:
                if output[i] == actual[i]:
                    self.stime_tp += 1

            except:
                continue

    def process_etime_tps(self, actual, output):
        self.etime_classified += len(output)
        self.etime_true_count += len(actual)

        for i in range(0, len(output)):
            try:
                if output[i] == actual[i]:
                    self.etime_tp += 1

            except:
                continue
    def process_sents_tps(self, actual, output):
        self.sentence_classified += len(output)
        self.sentence_true_count += len(actual)

        for i in range(0, len(output)):
            try:
                if output[i] == actual[i]:
                    self.sentence_tp += 1

            except:
                continue

    def process_para_tps(self, actual, output):
        self.paragraph_classified += len(output)
        self.paragraph_true_count += len(actual)

        for i in range(0, len(output)):
            try:
                if output[i] == actual[i]:
                    self.paragraph += 1

            except:
                continue

    def calc_precision(self, tps, classified):
        return tps/classified

    def calc_recall(self, tps, actual_tp_count):
        return tps/actual_tp_count

    def run(self):
        directory = os.fsencode('/Users/Adam/Documents/BRUM/SecondYear/Modules/NLP/Assignment/data/seminar_testdata/test_tagged/')
        for file in os.listdir(directory):
            try:
                filename = os.fsdecode(file)
                if filename.endswith(".txt"):
                    # print(filename)
                    actual_speakers, actual_locations, actual_stimes, actual_etimes, actual_sents, actual_paras = self.extractTestData(
                        filename)
                    out_speakers, out_locations, out_stimes, out_etimes, out_sents, out_paras = self.extractOutData(
                        filename)
                    self.process_etime_tps(actual_etimes, out_etimes)
                    self.process_loc_tps(actual_locations, out_locations)
                    self.process_stime_tps(actual_stimes, out_stimes)
                    self.process_sents_tps(actual_sents, out_sents)
                    self.process_para_tps(actual_paras, out_paras)
                    self.process_speaker_tps(actual_speakers, out_speakers)

                    continue
                else:
                    continue
            except:
                continue

        stime_precision = self.calc_precision(self.stime_tp, self.stime_classified)
        stime_recall = self.calc_recall(self.stime_tp, self.stime_true_count)

        etime_precision = self.calc_precision(self.etime_tp, self.etime_classified)
        etime_recall = self.calc_recall(self.etime_tp, self.etime_true_count)

        location_precision = self.calc_precision(self.loc_tp, self.loc_classified)
        location_recall =  self.calc_recall(self.loc_tp, self.loc_true_count)


        speaker_precision = self.calc_precision(self.speaker_tp, self.speaker_classified)
        speaker_recall = self.calc_recall(self.speaker_tp, self.speaker_true_count)


        sent_precision = self.calc_precision(self.sentence_tp, self.sentence_classified)
        para_precision = self.calc_precision(self.paragraph_tp, self.paragraph_classified)

        print('stime precision is: {}'.format(stime_precision))
        print('stime recall is: {}'.format(stime_recall))
        print('etime precision is: {}'.format(etime_precision))
        print('etime recall is: {}'.format(etime_recall))
        print('location precison is: {}'.format(location_precision))
        print('location recall is: {}'.format(location_recall))

        print('speaker precision is: {}'.format(speaker_precision))
        print('speaker recall is: {}'.format(speaker_recall))
        # print(sent_precision)
        # print(para_precision)

        # print(self.etime_tp)

if __name__ == '__main__':
    eval = Evaluation()
    # aspeakers, alocations, astimes, aetimes, asents, aparas = eval.extractTestData('401.txt')
    # speakers, locations, stimes, etimes, sents, paras = eval.extractOutData('401.txt')
    #
    # print(sents)
    # print(asents)
    # eval.process_sents_tps(asents, sents)
    # print(eval.calc_precision(eval.sentence_tp, eval.sentence_classified))

    # eval.extractOutData('401.txt')

    eval.run()
