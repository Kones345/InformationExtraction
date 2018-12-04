import re
import spacy
import itertools
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim
from gensim import corpora
from nltk.stem.wordnet import WordNetLemmatizer
import string
import os
from pathlib import Path

# path = '/Users/Adam/Documents/BRUM/SecondYear/Modules/NLP/Assignment/data/untagged/'

# stop_words = set(stopwords.words('english'))
# stop_words.add('lecture')
# stop_words.add('seminar')
# exclude = set(string.punctuation)

# special_char_regx_str = r'([^a-zA-Z ]+?)'
# cleanTextRegex = re.compile(special_char_regx_str)

# def clean(doc):
# 	stop_free = " ".join([i for i in doc.lower().split() if i not in stop_words])
# 	punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
# 	normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())

# 	return normalized


# from regex_store import *
# documents = []
# untagged = 'data/untagged/'
# pathlist = Path(untagged).glob('**/*.txt')
# for path in pathlist:
# 	p = str(path)
# 	with open(p, 'r', encoding='utf-8') as f:
# 		text = f.read()
# 		#Splits the text into header and body
# 		try:
# 			header, body = re.search(header_body_regx_str, text).groups()
# 		except:

# 			continue
# 		body = re.sub(cleanTextRegex, '',body)
# 		documents.append(' '.join(body.split()))

# doc_clean = [clean(doc).split() for doc in documents] 

# # print(doc_clean)

# dictionary = corpora.Dictionary(doc_clean)
# print(dictionary)
# # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
# doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# # Creating the object for LDA model using gensim library
# Lda = gensim.models.ldamodel.LdaModel

# # Running and Trainign LDA model on the document term matrix.
# ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)

# print(ldamodel.print_topics(num_topics=10, num_words=10))
data = {
    'Computer Science': ['software', 'object-oriented', 'architecture', 'design', 'product', 'debug', 'breakpoint',
                         'planning', 'a*', 'd*'
                                           'development', 'quality', 'tests', 'artificial', 'intelligence', 'recursion',
                         'iterative',
                         'machine learning', 'ai', 'robotics', 'vision', 'data', 'complexity', 'search',
                         'nlp', 'natural language processing', 'ieee', 'navigation', 'robot', 'planning',
                         'humanoid', 'autonomous', 'knowledge', 'language', 'decision', 'recognition',
                         'classification', 'prediction', 'human', 'user', 'friendly', 'interaction', 'graphics',
                         'fun', 'usable', 'design', 'hci', 'interface', 'style', 'communication', 'visual',
                         'render', 'parallel', 'distributed', 'network', 'synchronization', 'efficient',
                         'synchronous', 'asynchronous', 'thread', 'multi', 'breach', 'cryptography', 'backdoor',
                         'encryption', 'decryption', 'hacking', 'algorithm', 'code', 'programming', 'java', 'python',
                         'ruby'],
    'Biology': ['bio', 'disease', 'immune', 'immunomodulation', 'biological', 'genome', 'biochemistry', 'molecules',
                'medicine', 'clinic', 'cancer', 'health', 'cellular', 'cells', 'respiration', 'brain', 'neurological'],
    'Chemistry': ['chemistry', 'drugs', 'polymers', 'graft', 'extruders', 'nanotechnology', 'fluids', 'thermodynamic',
                  'microemulsions', 'flourescence', 'water', 'dissolved', 'exothermic', 'endothermic', 'alcohol'],
    'Electronics': ['semiconductor', 'electronic', 'circuit', 'integrated', 'multimeter', 'voltage', 'controller'],
    'Physics': ['physics', 'thermodynamics', 'nanotechnology', 'magnetism', 'frequency', 'nuclear', 'stars',
                'cosmology', 'black', 'hole',
                'resistance', 'voltage', 'current', 'mechanics', 'momentum', 'velocity', 'acceleration', 'particles',
                'electricity',
                'fields', 'astrophysics', 'photon', 'radiation', 'electromagnet', 'lepton', 'diffraction',
                'interference', 'motion',
                'projectile', 'gravity', 'telescope', 'Quasars'],
    'Politics': ['politics', 'social', 'public', 'policy', 'issue', 'environment', 'trend',
                 'economy', 'media', 'global', 'regulations', 'international', 'crisis',
                 'activist', 'prospects', 'welfare', 'community', 'movement', 'liberal', 'republican',
                 'democracy', 'communist', 'left', 'right', 'centre', 'totalitarian', 'dictatorship',
                 'vote', 'election', 'rally', 'ballot'],
    'Languages': ['language', 'phonology', 'english', 'writing', 'french', 'spanish', 'german', 'finnish',
                  'chinese', 'japanese', 'dictionary', 'lexicon', 'dialect', 'reading', 'listening', 'comprehension',
                  'travel', 'tourist', 'learning', 'bilingual', 'fluent', 'native'],
    'Performing Arts': ['music', 'genre', 'ensemble', 'classical', 'theater', 'actor', 'actress', 'film', 'musical',
                        'performance', 'audition',
                        'notes', 'instrument', 'orchestra', 'chopin', 'mozart', 'symphony', 'tickets', 'production',
                        'stage', 'cinema', 'movie',
                        'play'],
    'Business': ['entrepreneurship', 'innovation', 'opportunities', 'customer',
                 'finance', 'stock', 'exchange', 'money', 'investment', 'trader', 'bank', 'trend',
                 'interest', 'business', 'businessman', 'businesswoman', 'market', 'need', 'global', 'success',
                 'office', 'economy', 'management', 'equity', 'competition', 'revenue', 'profit'],
    'Education': ['education', 'academia', 'alumni', 'teaching', 'school', 'graduate', 'academic', 'grade', 'degree',
                  'university', 'college', 'school', 'nursery', 'homework', 'assignment', 'dissertation', 'doctorate',
                  'masters', 'phd', 'research']

}

# Words -> category
categories = {word: key for key, words in data.items() for word in words}

# Load the whole embedding matrix
embeddings_index = {}
with open('glove.6B.100d.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        embed = np.array(values[1:], dtype=np.float32)
        embeddings_index[word] = embed
print('Loaded %s word vectors.' % len(embeddings_index))
# Embeddings for available words
data_embeddings = {key: value for key, value in embeddings_index.items() if key in categories.keys()}


# Processing the query
def process(query):
    scores = {}

    try:
        query_embed = embeddings_index[query]
    except:
        for c in categories:
            scores[c] = 0
        return scores

    for word, embed in data_embeddings.items():
        category = categories[word]
        dist = query_embed.dot(embed)
        dist /= len(data[category])
        scores[category] = scores.get(category, 0) + dist
    return scores


stop_words = set(stopwords.words('english'))
stop_words.add('lecture')
stop_words.add('seminar')
fileid = '409'
fileLoc = '/Users/Adam/Documents/BRUM/SecondYear/Modules/NLP/Assignment/data/untagged/' + fileid + '.txt'
directory = '/Users/Adam/Documents/BRUM/SecondYear/Modules/NLP/Assignment/data/untagged/'
topic_regx_str = r'(?:\b(?:Topic)\b:\s*)(.*)'
special_char_regx_str = r'([^a-zA-Z ]+?)'

lemma = WordNetLemmatizer()
topicRegex = re.compile(topic_regx_str)
cleanTextRegex = re.compile(special_char_regx_str)

pathlist = Path(directory).glob('**/*.txt')
for path in pathlist:
    p = str(path)
    with open(p, 'r', encoding='utf-8') as f:

        print(p + '\n')
        content = f.read()
        topic = re.search(topicRegex, content)
        if topic is None:
            continue
        topic = topic.group(1).lower()
        topic = re.sub(cleanTextRegex, '', topic)
        normalized = " ".join(lemma.lemmatize(word) for word in topic.split())
        word_tokens = word_tokenize(normalized)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]

        for word in filtered_sentence:
            print('Result for %s: ' % word)
            print(process(word))
            print()

# print(process('robot'))
