from nltk.tag import UnigramTagger
from nltk.tag import BigramTagger
from nltk.tag import TrigramTagger
from nltk.tag import DefaultTagger
from nltk.corpus import brown
import nltk
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
            '/Users/Adam/Documents/BRUM/SecondYear/Modules/NLP/Assignment/stanfordNERJars/classifiers/english.all.3class.distsim.crf.ser.gz',
            '/Users/Adam/Documents/BRUM/SecondYear/Modules/NLP/Assignment/stanfordNERJars/stanford-ner.jar',
            encoding='utf-8')
        if os.path.exists("out/"):
            shutil.rmtree('out/')

    train_sents = brown.tagged_sents()[:48000]

    def backoff_tagger(self, backoff=None):

        for cls in [UnigramTagger, BigramTagger, TrigramTagger]:
            backoff = cls(self.train_sents, backoff=backoff)
        return backoff

    def tagPOS(self, body):

        sents = nltk.tokenize.sent_tokenize(body)
        for i in range(0, len(sents)):
            sents[i] = nltk.tokenize.word_tokenize(sents[i])

        tagged = self.backoff.tag_sents(sents)
        flatten = lambda l: [item for sublist in l for item in sublist]
        tagged = flatten(tagged)
        processed_body = ['{}{{*{}*}}'.format(word, tag) for (word, tag) in tagged]

        return ' '.join(processed_body)

    def nerStanford(self, text, entitiy):
        tokenized_text = word_tokenize(text)
        classified_text = self.st.tag(tokenized_text)
        results = []
        for tag, chunk in groupby(classified_text, lambda x: x[1]):
            if tag == entitiy:
                results.append(" ".join(w for w, t in chunk))
                # print( " ".join(w for w, t in chunk))
        return set(results)

    @staticmethod
    def split_on_tags(text, tag):
        return re.split(r'</?{}>'.format(tag), text)

    @staticmethod
    def tag_paragraphs(text):
        text = '\n\n{}\n\n'.format(text.strip('\n'))
        para = re.compile(paragraphRegex)
        for match in para.finditer(text):
            paragraph = match.group(1)
            if paragraph:
                text = text.replace(paragraph, '<paragraph>{}</paragraph>'.format(paragraph))

        return text.strip()

    def tag_sentences(self, text):
        text_parts = self.split_on_tags(text, 'paragraph')
        sentences = []
        for part in text_parts:
            sentences.extend(sent_tokenize(part.strip()))

        # filter everything that is not a proper sentence
        sentences = list(filter(lambda s: re.match(not_sentence_regx_str, s), sentences))
        for sent in sentences:
            text = text.replace(sent, '<sentence>{}</sentence>'.format(sent))

        return text

    @staticmethod
    def tagTimes(stime, etime, text):

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
        for loc in locations:
            insensitive_loc = re.compile(r'({})'.format(re.escape(loc)), re.IGNORECASE)
            text = re.sub(insensitive_loc, '<location>' + loc + '</location>', text)

        return text

    @staticmethod
    def tag_speakers(text, speakers):

        for spk in speakers:

            insensitive_spk = re.compile(r'(\b({})\b|[.?!]({})\b|\(({})\))'.format(re.escape(spk), re.escape(spk),
                                                                                   re.escape(spk), re.escape(spk)),
                                         re.IGNORECASE)
            try:
                name = re.search(insensitive_spk, text).group(1)
                clean = name.strip()
                text = text.replace(name, '<speaker>' + clean + '</speaker>')
            except:
                continue

        return text

    def tagSeminar(self, path, directory, extractor):
        for file in tqdm(os.listdir(directory)):
            filename = os.fsdecode(file)
            if filename.endswith(".txt"):
                with open(path + filename, 'r', encoding='utf-8') as f:
                    placeholder = f.read()

                    # Splits the text into header and body
                    try:
                        header, body = re.search(header_body_regx_str, placeholder).groups()
                    except:
                        continue

                    stime, etime = extractor.extractTime(header)
                    locations = extractor.extractLocation(header, body, self)
                    speakers = extractor.extractSpeaker(header, body, self)

                    body = self.tag_paragraphs(body)
                    body = self.tag_sentences(body)

                    seminar = header + '\n\n' + body
                    seminar = self.tagTimes(stime, etime, seminar)
                    seminar = self.tag_speakers(seminar, speakers)
                    seminar = self.tag_locations(locations, seminar)

                    out_location = "out/"
                    Utils.mkdir_p(out_location)
                    out = open(out_location + filename, "w+")
                    out.write(seminar)
                    out.close()
                    # print(filename)
                continue
