import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import re
from regex_store import *
import os
from tabulate import tabulate


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

    @staticmethod
    def extract_test_data(filename):
        """
        Extracts tagged data from test files
        :param filename: name of file to extract from
        :return: the tagged data
        """
        test_file_path = '/Users/Adam/Documents/BRUM/SecondYear/Modules/NLP/Assignment/data/seminar_testdata/test_tagged/'
        file = open(test_file_path + filename, 'r')
        contents = file.read()
        speaker_regex = re.compile(speaker_regex_str)
        speakers = speaker_regex.findall(contents)
        location_regex = re.compile(knownLocationRegxStr)
        locations = re.findall(location_regex, contents)
        stimes = re.findall(stime_regex_str, contents)
        stimes = [s.replace(':', '') for s in stimes]
        stimes = [s.replace('.', '') for s in stimes]
        stimes = [s.replace(' ', '') for s in stimes]
        etimes = re.findall(etime_regex_str, contents)
        etimes = [e.replace(':', '') for e in etimes]
        etimes = [e.replace('.', '') for e in etimes]
        etimes = [e.replace(' ', '') for e in etimes]
        sents = re.findall(sent_regex_str, contents)
        sents = [s.replace('.', '') for s in sents]
        sents = [s.replace(' ', '') for s in sents]
        sents = [re.sub('<\/?[a-z]+?>', '', s) for s in sents]
        paras = re.findall(para_regex_str, contents)
        paras = [re.sub('<\/?[a-z]+?>', '', p) for p in paras]
        paras = [p.replace(' ', '') for p in paras]

        # print(speakers)
        # print(locations)
        # print(stimes)
        # print(etimes)
        # print(sents)
        # print(paras)
        return speakers, locations, stimes, etimes, sents, paras

    @staticmethod
    def extract_out_data(filename):
        """
        Extracts the output data from the files I tagged
        :param filename: the file to have data extracted from
        :return: the data extracted
        """
        test_file_path = '/Users/Adam/Documents/BRUM/SecondYear/Modules/NLP/Assignment/out/'
        file = open(test_file_path + filename, 'r')
        contents = file.read()
        speaker_regex = re.compile(speaker_regex_str)
        speakers = speaker_regex.findall(contents)
        location_regex = re.compile(knownLocationRegxStr)
        locations = re.findall(location_regex, contents)
        stimes = re.findall(stime_regex_str, contents)
        stimes = [s.replace(':', '') for s in stimes]
        stimes = [s.replace('.', '') for s in stimes]
        stimes = [s.replace(' ', '') for s in stimes]
        etimes = re.findall(etime_regex_str, contents)
        etimes = [e.replace(':', '') for e in etimes]
        etimes = [e.replace('.', '') for e in etimes]
        etimes = [e.replace(' ', '') for e in etimes]
        sents = re.findall(sent_regex_str, contents)
        sents = [s.replace('.', '') for s in sents]
        sents = [s.replace(' ', '') for s in sents]
        sents = [re.sub('<\/?[a-z]+?>', '', s) for s in sents]

        paras = re.findall(para_regex_str, contents)
        paras = [p.replace(' ', '') for p in paras]
        paras = [re.sub('<\/?[a-z]+?>', '', p) for p in paras]

        # print(speakers)
        # print(locations)
        # print(stimes)
        # print(etimes)
        # print(sents)
        # print(paras)
        return speakers, locations, stimes, etimes, sents, paras

    def process_loc_tps(self, actual, output):
        """
        Processes the true positives for location
        :param actual: actual locations from test data
        :param output: data from output files
        """
        self.loc_classified += len(output)
        self.loc_true_count += len(actual)

        for i in range(0, len(output)):
            try:
                if output[i] == actual[i]:
                    self.loc_tp += 1

            except:
                continue

    def process_speaker_tps(self, actual, output):
        """
        Processes the true positives for speakers
        :param actual: actual speakers from test data
        :param output: data from output files
        """
        self.speaker_classified += len(output)
        self.speaker_true_count += len(actual)

        for i in range(0, len(output)):
            try:
                if output[i] == actual[i]:
                    self.speaker_tp += 1

            except:
                continue

    def process_stime_tps(self, actual, output):
        """
        Processes the true positives for stimes
        :param actual: actual stimes from test data
        :param output: data from output files
        """
        self.stime_classified += len(output)
        self.stime_true_count += len(actual)

        for i in range(0, len(output)):
            try:
                if output[i] == actual[i]:
                    self.stime_tp += 1

            except:
                continue

    def process_etime_tps(self, actual, output):
        """
        Processes the true positives for etimes
        :param actual: actual etimes from test data
        :param output: data from output files
        """
        self.etime_classified += len(output)
        self.etime_true_count += len(actual)

        for i in range(0, len(output)):
            try:
                if output[i] == actual[i]:
                    self.etime_tp += 1

            except:
                continue

    def process_sents_tps(self, actual, output):
        """
        Processes the true positives for sents
        :param actual: actual sentences from test data
        :param output: data from output files
        """
        self.sentence_classified += len(output)
        self.sentence_true_count += len(actual)

        for i in range(0, len(output)):
            try:
                for j in range(0, len(actual)):
                    if output[i] == actual[j]:
                        self.sentence_tp += 1

            except:
                continue

    def process_para_tps(self, actual, output):
        """
        Processes the true positives for paragraphs
        :param actual: actual paragraphs from test data
        :param output: data from output files
        """
        self.paragraph_classified += len(output)
        self.paragraph_true_count += len(actual)

        for i in range(0, len(output)):
            try:
                for j in range(0, len(actual)):
                    if output[i] == actual[j]:
                        self.paragraph_tp += 1
            except:
                continue

    @staticmethod
    def calc_precision(tps, classified):
        """
        Calculates precision given true positives and total amount classified
        :param tps: the number of true positives
        :param classified: the number of a variable classified
        :return: the precision
        """
        return tps / classified

    @staticmethod
    def calc_recall(tps, actual_tp_count):
        """
        Calculates the recall.
        :param tps: the number of true positives found
        :param actual_tp_count: the number of true positives
        :return: the recall value
        """
        return tps / actual_tp_count

    @staticmethod
    def calc_f_measure(precision, recall):
        """
        The F1 measure
        :param precision: the precision of a variable
        :param recall: the recall of the variable
        :return: the f measure value
        """
        return ((precision * recall) / (precision + recall)) * 2

    def run(self):
        """
        Processes precision, recall and f measure for each of the tags and prints the results in a table
        """
        directory = os.fsencode(
            '/Users/Adam/Documents/BRUM/SecondYear/Modules/NLP/Assignment/data/seminar_testdata/test_tagged/')
        for file in os.listdir(directory):
            try:
                filename = os.fsdecode(file)
                if filename.endswith(".txt"):
                    actual_speakers, actual_locations, actual_stimes, actual_etimes, actual_sents, \
                                                                        actual_paras = self.extract_test_data(filename)
                    out_speakers, out_locations, out_stimes, out_etimes, out_sents, out_paras = self.extract_out_data(
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
        stime_f = self.calc_f_measure(stime_precision, stime_recall)

        stime_row = ['stime', stime_precision, stime_recall, stime_f]

        etime_precision = self.calc_precision(self.etime_tp, self.etime_classified)
        etime_recall = self.calc_recall(self.etime_tp, self.etime_true_count)
        etime_f = self.calc_f_measure(etime_precision, etime_recall)
        etime_row = ['etime', etime_precision, etime_recall, etime_f]

        location_precision = self.calc_precision(self.loc_tp, self.loc_classified)
        location_recall = self.calc_recall(self.loc_tp, self.loc_true_count)
        location_f = self.calc_f_measure(location_precision, location_recall)
        loc_row = ['location', location_precision, location_recall, location_f]

        speaker_precision = self.calc_precision(self.speaker_tp, self.speaker_classified)
        speaker_recall = self.calc_recall(self.speaker_tp, self.speaker_true_count)
        speaker_f = self.calc_f_measure(speaker_precision, speaker_recall)
        speaker_row = ['speaker', speaker_precision, speaker_recall, speaker_f]

        sent_precision = self.calc_precision(self.sentence_tp, self.sentence_classified)
        sent_recall = self.calc_recall(self.sentence_tp, self.sentence_true_count)
        sent_f = self.calc_f_measure(sent_precision, sent_recall)
        sent_row = ['sentence', sent_precision, sent_recall, sent_f]

        para_precision = self.calc_precision(self.paragraph_tp, self.paragraph_classified)
        para_recall = self.calc_recall(self.paragraph_tp, self.paragraph_true_count)
        para_f = self.calc_f_measure(para_precision, para_recall)
        para_row = ['paragraph', para_precision, para_recall, para_f]
        print(tabulate([stime_row, etime_row, loc_row, speaker_row, sent_row, para_row],
                       headers=['Tag', 'Precision', 'Recall', 'F Measure']))


if __name__ == '__main__':
    e = Evaluation()
    # eval.extractOutData('301.txt')
    # eval.extractTestData('301.txt')
    e.run()
