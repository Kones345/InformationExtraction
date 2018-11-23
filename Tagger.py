from nltk.tag import UnigramTagger
from nltk.tag import BigramTagger
from nltk.tag import TrigramTagger
from nltk.tag import DefaultTagger
from nltk.corpus import brown
import nltk
# from nltk.tag.stanford import NERTagger
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tag import StanfordNERTagger
from itertools import groupby

class Tagger():
    
    # backoff

    def __init__(self):
        self.backoff =  self.backoff_tagger(backoff=DefaultTagger('NN'))
        self.st = StanfordNERTagger(
            '/Users/Adam/Documents/BRUM/SecondYear/Modules/NLP/Assignment/stanfordNERJars/classifiers/english.all.3class.distsim.crf.ser.gz',
            '/Users/Adam/Documents/BRUM/SecondYear/Modules/NLP/Assignment/stanfordNERJars/stanford-ner.jar',
         encoding='utf-8')
    # test_sents = brown.tagged_sents()[48000:]
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
    
    