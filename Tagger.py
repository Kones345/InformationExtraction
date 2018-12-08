import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from nltk.tag import UnigramTagger
from nltk.tag import BigramTagger
from nltk.tag import TrigramTagger
from nltk.tag import DefaultTagger
from nltk.corpus import brown
from dateutil import parser as time_parser
from nltk import word_tokenize, sent_tokenize
from nltk.tag import StanfordNERTagger
from itertools import groupby
from regex_store import *
import re
from Utils import Utils
import os
from tqdm import tqdm
import shutil


# Class to handle tagging of seminar emails
class Tagger:

    def __init__(self):
        self.backoff = self.backoff_tagger(backoff=DefaultTagger('NN'))
        self.st = StanfordNERTagger(
            '/Users/Adam/Documents/BRUM/SecondYear/Modules/NLP/Assignment/stanfordNERJars/classifiers/english.all'
            '.3class.distsim.crf.ser.gz',
            '/Users/Adam/Documents/BRUM/SecondYear/Modules/NLP/Assignment/stanfordNERJars/stanford-ner.jar',
            encoding='utf-8')
        if os.path.exists("out/"):
            shutil.rmtree('out/')

    train_sents = brown.tagged_sents()[:48000]

    def backoff_tagger(self, backoff=None):
        """
        Used to tag text using a more accurate backoff tagger
        :param backoff: the current backoff
        :return: a backoff tagger
        """
        for cls in [UnigramTagger, BigramTagger, TrigramTagger]:
            backoff = cls(self.train_sents, backoff=backoff)
        return backoff

    def ner_stanford(self, text, entity):
        """
        Gets a list of specific entities from text
        :param text: the text we want to search in
        :param entity: the entity to extract
        :return: the list of entities
        """
        tokenized_text = word_tokenize(text)
        classified_text = self.st.tag(tokenized_text)
        results = []
        for tag, chunk in groupby(classified_text, lambda x: x[1]):
            if tag == entity:
                results.append(" ".join(w for w, t in chunk))
        return set(results)


    @staticmethod
    def tag_paragraphs(text):
        """
        Tags paragraphs in text
        :param text: text to be tagged
        :return:
        """
        text = '\n\n{}\n\n'.format(text.strip('\n'))
        para = re.compile(paragraphRegex)
        for match in para.finditer(text):
            paragraph = match.group(1)
            if paragraph:
                text = text.replace(paragraph, '<paragraph>{}</paragraph>'.format(paragraph))

        return text.strip()

    def tag_sentences(self, text):
        """
        Tags sentences in the text
        :param text: text to be tagged
        :return: tagged text
        """
        # text_parts = self.split_on_tags(text, 'paragraph')
        text_parts = re.split(r'</?{}>'.format('paragraph'), text)
        sentences = []
        for part in text_parts:
            p = part.strip()
            s = sent_tokenize(p)
            sentences.extend(s)
            # sentences.extend(sent_tokenize(part.strip()))

        # filter everything that is not a proper sentence
        temp = []
        for sent in sentences:
            res = re.match(not_sentence_regx_str, sent)
            if res is not None:
                temp.append(sent)

        # sentences = list(filter(lambda s: re.match(not_sentence_regx_str, s), sentences))
        for sent in temp:
            text = text.replace(sent, '<sentence>{}</sentence>'.format(sent))

        return text

    @staticmethod
    def tag_times(stime, etime, text):
        """
        Tags times in the text
        :param stime: the start time
        :param etime: the end time
        :param text: the text to tag
        :return: the text tagged with times
        """
        if not etime and not stime:
            return text
        textHolder = text
        time_regx = re.compile(time_regx_str)

        for time_str in set(time_regx.findall(textHolder)):
            time = time_parser.parse(time_str).time()
            if time_parser.parse(stime).time() == time:
                textHolder = textHolder.replace(time_str, '<stime>{}</stime>'.format(time_str))

            elif etime:
                if time_parser.parse(etime).time() == time:
                    textHolder = textHolder.replace(time_str, '<etime>{}</etime>'.format(time_str))
        return textHolder

    @staticmethod
    def tag_locations(locations, text):
        """
        Tags locations in the text
        :param locations: locations to be tagged
        :param text: text to be tagged
        :return: the text with locations tagged
        """
        for loc in locations:
            compiled = re.compile(re.escape(loc), flags=re.IGNORECASE)
            text = re.sub(compiled, '<location>' + loc + '</location>', text)

        return text

    @staticmethod
    def tag_speakers(text, speakers):
        """
        Tags speakers in the text
        :param text: text to tag
        :param speakers: speakers to tag
        :return: the tagged text
        """
        for spk in speakers:

            insensitive_spk = re.compile(r'(\b({})\b|[.?!]({})\b|\(({})\))'.format(re.escape(spk), re.escape(spk),
                                                                                   re.escape(spk), re.escape(spk)),
                                         re.IGNORECASE)
            try:
                name = re.search(insensitive_spk, text).group(1)
                clean = name.strip()
                text = text.replace(name, '<speaker>' + clean + '</speaker>')
            except:
                pass

        return text

    def tag_seminar(self, path, directory, extractor):
        """
        Tags seminar with all previously found data and writes the data to a file.
        :param path: the path to the untagged files
        :param directory: the directory they are in
        :param extractor: the extractor class to extract data
        """
        for file in tqdm(os.listdir(directory)):
            filename = os.fsdecode(file)
            if filename.endswith(".txt"):
                with open(path + filename, 'r', encoding='utf-8') as f:
                    placeholder = f.read().strip('\n -*')

                    # Splits the text into header and body
                    try:
                        header, body = re.search(header_body_regx_str, placeholder).groups()
                    except:
                        continue

                    header = header.rstrip('\n')

                    stime, etime = extractor.extract_time(header)
                    locations = extractor.extract_location(header, body, self)
                    speakers = extractor.extract_speaker(header, body, self)

                    body = self.tag_paragraphs(body)
                    body = self.tag_sentences(body)

                    seminar = header + '\n\n' + body
                    seminar = self.tag_times(stime, etime, seminar)
                    seminar = self.tag_speakers(seminar, speakers)
                    seminar = self.tag_locations(locations, seminar)

                    out_location = "out/"
                    Utils.mkdir_p(out_location)
                    out = open(out_location + filename, "w+")
                    out.write(seminar)
                    out.close()
                continue
