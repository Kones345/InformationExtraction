from nltk.tag import UnigramTagger
from nltk.tag import BigramTagger
from nltk.tag import TrigramTagger
from nltk.tag import DefaultTagger
from nltk.corpus import brown
import nltk
from dateutil import parser as time_parser
# from nltk.tag.stanford import NERTagger
from nltk import ne_chunk, pos_tag, word_tokenize, sent_tokenize
from nltk.tag import StanfordNERTagger
from itertools import groupby
from regex_store import *
import re

#Class to handle tagging of seminar emails
class Tagger():
    
    # backoff
    def __init__(self):
        self.backoff =  self.backoff_tagger(backoff=DefaultTagger('NN'))
        self.st = StanfordNERTagger(
            '/Users/Adam/Documents/BRUM/SecondYear/Modules/NLP/Assignment/stanfordNERJars/classifiers/english.all.3class.distsim.crf.ser.gz',
            '/Users/Adam/Documents/BRUM/SecondYear/Modules/NLP/Assignment/stanfordNERJars/stanford-ner.jar',
         encoding='utf-8')

    train_sents = brown.tagged_sents()[:48000]
    def backoff_tagger(self, backoff=None):
        
        for cls in [UnigramTagger, BigramTagger, TrigramTagger] :
            backoff = cls(self.train_sents, backoff=backoff)
        return backoff
    
    def tagPOS(self,body):
        
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
        for tag, chunk in groupby(classified_text, lambda x:x[1]):
            if tag == entitiy:
                results.append(" ".join(w for w, t in chunk))
                # print( " ".join(w for w, t in chunk))
        return set(results)
    
    def split_on_tags(self,text, tag):
        return re.split(r'</?{}>'.format(tag), text)

    def tag_paragraphs(self, text):
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
    
    def tagTimes(self, stime, etime, text):

        if not etime and not stime:
            return text
        textHolder = text
        time_regx = re.compile(time_regx_str)

        for time_str in set(time_regx.findall(textHolder)):
            time = time_parser.parse(time_str).time()
            if time_parser.parse(stime).time() == time:
                textHolder = re.sub(time_str, '<stime>{}</stime>'.format(stime), textHolder)

            elif etime:
                if time_parser.parse(etime).time() == time:
                    textHolder = re.sub(time_str, '<etime>{}</etime>'.format(etime), textHolder)

        
        return textHolder
    
    def tag_locations(self, locations, text):
        for loc in locations:
            insensitive_loc = re.compile(r'({})'.format(re.escape(loc)), re.IGNORECASE)
            text = re.sub(insensitive_loc, '<location> '+ loc + '</location>', text)

        return text

    def tag_speakers(self, text, speakers):
        for spk in speakers:
            insensitive_spk = re.compile(r'({})'.format(re.escape(spk)))
            text = re.sub(insensitive_spk, r'<speaker>\1</speaker>', text)

        return text

