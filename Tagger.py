from nltk.tag import UnigramTagger
from nltk.tag import BigramTagger
from nltk.tag import TrigramTagger
from nltk.tag import DefaultTagger
from nltk.corpus import brown
import nltk

class Tagger():
    
    # backoff

    def __init__(self):
        self.backoff =  self.backoff_tagger(backoff=DefaultTagger('NN'))
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