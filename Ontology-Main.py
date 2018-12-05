import re

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from pathlib import Path

data = {
    'Computer Science': ['software', 'object-oriented', 'architecture', 'design', 'product', 'debug', 'breakpoint',
                         'planning', 'a*', 'd*', 'development', 'quality', 'tests', 'artificial', 'intelligence',
                         'recursion',
                         'iterative',
                         'machine learning', 'ai', 'robotics', 'vision', 'data', 'complexity', 'search',
                         'nlp', 'natural language processing', 'ieee', 'navigation', 'robot', 'planning',
                         'humanoid', 'autonomous', 'knowledge', 'language', 'decision', 'recognition',
                         'classification', 'prediction', 'human', 'user', 'friendly', 'interaction', 'graphics',
                         'fun', 'usable', 'design', 'hci', 'interface', 'style', 'communication', 'visual',
                         'render', 'parallel', 'distributed', 'network', 'synchronization', 'efficient',
                         'synchronous', 'asynchronous', 'thread', 'multi', 'breach', 'cryptography', 'backdoor',
                         'encryption', 'decryption', 'hacking', 'algorithm', 'code', 'programming', 'java', 'python',
                         'ruby', 'robotics', 'circuit', 'computing', 'systems', 'framework'],
    'Biology': ['bio', 'disease', 'immune', 'immunonoculation', 'biological', 'genome', 'biochemistry', 'molecules',
                'medicine', 'clinic', 'cancer', 'health', 'cellular', 'cells', 'respiration', 'brain', 'neurological',
                'imaging', 'illness', 'sick', 'disease', 'healthcare'],
    'Chemistry': ['chemistry', 'drugs', 'polymers', 'graft', 'extruders', 'nanotechnology', 'fluids', 'thermodynamic',
                  'microemulsions', 'flourescence', 'water', 'dissolved', 'exothermic', 'endothermic', 'alcohol'],
    'Physics and Astronomy': ['physics', 'thermodynamics', 'nanotechnology', 'magnetism', 'frequency', 'nuclear',
                              'stars',
                              'cosmology', 'black', 'hole', 'resistance', 'voltage', 'current', 'mechanics', 'momentum',
                              'velocity',
                              'acceleration', 'particles', 'electricity', 'fields', 'astrophysics', 'photon',
                              'radiation', 'electromagnet', 'lepton', 'diffraction',
                              'interference', 'motion',
                              'projectile', 'gravity', 'telescope', 'Quasars', 'semiconductor', 'electronic', 'circuit',
                              'integrated', 'multimeter', 'voltage', 'controller'],
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
                  'masters', 'phd', 'research'],
    'Engineering': ['material', 'engineering', 'environment', 'design', '3d', 'aeroacoustics', 'aerothermodynamics',
                    'air', 'turbulence', 'bridge', 'collapse', 'building', 'cad', 'car', 'fiber', 'optics',
                    'geoengineering', 'nano', 'nanotubes', 'stereo', 'tunnel'],
    'Mathematics': ['anlge', 'measure', 'prove', 'solve', 'problem', 'equation', 'graph', 'plane', 'line', 'axis', 'algebra',
                    'adjacent', 'coefficient', 'frequency', 'circumference', 'denominator', 'distribution', 'polygon',
                    'expression', 'factorise', 'formula', 'frequency', 'density', 'gradient', 'hcf', 'indices',
                    'integer', 'quartile', 'rational', 'irrational', 'lcm', 'average', 'mean', 'median', 'mode',
                    'numerator', 'even', 'odd', 'parallel', 'perpendicular', 'probability', 'product', 'prime',
                    'quadratic', 'remainder', 'rotation', 'rotate', 'sum', 'symmetry', 'tangent', 'volume',
                    'solve'],
    'Other': []

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
        return None

    for word, embed in data_embeddings.items():
        category = categories[word]
        dist = query_embed.dot(embed)
        dist /= len(data[category])
        scores[category] = scores.get(category, 0) + dist
    return scores


stop_words = set(stopwords.words('english'))
extra_stop_words = {'lecture', 'seminar', 'talk', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
                    'saturday', 'sunday', 'telecastseminar', 'today', 'weh', 'adamson', 'wing', 'hall', 'baker',
                    'rescheduled', 'am', 'pm', 'reminder', 'jan', 'feb', 'march', 'apr', 'may', 'jun', 'jul', 'aug',
                    'sep', 'oct', 'nov', 'dec', 'spring', 'summer', 'autumn', 'winter', 'moved', 'none'}

stop_words = stop_words.union(extra_stop_words)

fileid = '409'
fileLoc = '/Users/Adam/Documents/BRUM/SecondYear/Modules/NLP/Assignment/data/untagged/' + fileid + '.txt'
directory = '/Users/Adam/Documents/BRUM/SecondYear/Modules/NLP/Assignment/data/untagged/'
topic_regx_str = r'(?:\b(?:Topic)\b:\s*)(.*)'
special_char_regx_str = r'([^a-zA-Z ]+?)'

lemma = WordNetLemmatizer()
topicRegex = re.compile(topic_regx_str)
cleanTextRegex = re.compile(special_char_regx_str)

# ps = PorterStemmer()
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
        # filtered_sentence = [ps.stem(word) for word in filtered_sentence]

        results = {'Computer Science': 0, 'Politics': 0, 'Biology': 0, 'Chemistry': 0, 'Physics and Astronomy': 0,
                   'Languages': 0, 'Education': 0, 'Business': 0, 'Performing Arts': 0, 'Mathematics':0, 'Other':0, 'Engineering': 0}

        # print(nltk.pos_tag(filtered_sentence))
        tagged = nltk.pos_tag(filtered_sentence)
        filtered_sentence = [key for key, value in tagged if value == 'NN' or value == 'NNS']
        # print
        print(nltk.pos_tag(filtered_sentence))
        for word in filtered_sentence:
            # print('Result for %s: ' % word)
            resultStream = process(word)
            if resultStream is not None:

                for key, value in resultStream.items():
                    current_value = results[key]
                    current_value += value
                    results[key] = current_value
        print(results)
        print()
