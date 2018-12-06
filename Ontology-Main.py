import re

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from pathlib import Path
from collections import Counter

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
                         'ruby', 'robotics', 'circuit', 'computing', 'systems', 'framework', 'computer'],
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
    'Business': ['entrepreneurship', 'opportunities', 'customer',
                 'finance', 'stock', 'exchange', 'money', 'investment', 'trader', 'bank', 'trend',
                 'interest', 'business', 'businessman', 'businesswoman', 'market', 'need', 'global', 'success',
                 'office', 'economy', 'management', 'equity', 'competition', 'revenue', 'profit', ],
    'Education': ['education', 'academia', 'alumni', 'teaching', 'school', 'graduate', 'academic', 'grade', 'degree',
                  'university', 'college', 'school', 'nursery', 'homework', 'assignment', 'dissertation', 'doctorate',
                  'masters', 'phd', 'research'],
    'Engineering': ['material', 'engineering', 'environment', 'design', '3d', 'aeroacoustics', 'aerothermodynamics',
                    'air', 'turbulence', 'bridge', 'collapse', 'building', 'cad', 'car', 'fiber', 'optics',
                    'geoengineering', 'nano', 'nanotubes', 'stereo', 'tunnel'],
    'Mathematics': ['angle', 'measure', 'prove', 'solve', 'problem', 'equation', 'graph', 'plane', 'line', 'axis',
                    'algebra',
                    'adjacent', 'coefficient', 'frequency', 'circumference', 'denominator', 'distribution', 'polygon',
                    'expression', 'factorise', 'formula', 'frequency', 'density', 'gradient', 'hcf', 'indices',
                    'integer', 'quartile', 'rational', 'irrational', 'lcm', 'average', 'mean', 'median', 'mode',
                    'numerator', 'even', 'odd', 'parallel', 'perpendicular', 'probability', 'product', 'prime',
                    'quadratic', 'remainder', 'rotation', 'rotate', 'sum', 'symmetry', 'tangent', 'volume',
                    'solve', 'square', 'root', 'cube'],
    'Art': ['art', 'fine', 'color', 'colour', 'paint', 'gallery', 'print', 'pop', 'abstract', 'artist',
            'wall', 'acrylic', 'sculpture', 'watercolor', 'oil', 'supplies', 'modern', 'artwork', 'famous',
            'deco', 'pixel', '3d', 'liberal', 'vector', 'clip', 'creative', 'paper', 'nude', 'concept', 'animation'
            'crafts', 'clay', 'tattoo', 'pencil', 'graphic', 'portfolio', 'decor'],
    'Other': []

}

# Words -> category, makes a dictionary of keyword; category pairs
categories = {word: key for key, words in data.items() for word in words}

# Load the whole embedding matrix
embeddings_index = {}
with open('glove.6B.200d.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        embed = np.array(values[1:], dtype=np.float32)
        embeddings_index[word] = embed
print('Loaded %s word vectors.' % len(embeddings_index))
# Embeddings for available words -  stores all known keywords embeddings
data_embeddings = {key: value for key, value in embeddings_index.items() if key in categories.keys()}


# Processing the query
def process(query):
    scores = {}

    try:
        query_embed = embeddings_index[query]  # gets the embeddings for the word in question
    except:
        for c in categories:
            scores[c] = 0
        return None

    for word, embed in data_embeddings.items():  # iterates over the embeddings for known words
        category = categories[word]  # gets the category of keyword
        dist = query_embed.dot(embed)  # dot product of array of indexes for the current word and each keywords
        dist /= len(data[category])  # divides the dot product by the number of words in that category
        scores[category] = scores.get(category, 0) + dist  # Adds this calculated score to the current score in results
    return scores


stop_words = set(stopwords.words('english'))
extra_stop_words = {'lecture', 'seminar', 'talk', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
                    'saturday', 'sunday', 'telecastseminar', 'today', 'weh', 'adamson', 'wing', 'hall', 'baker',
                    'rescheduled', 'am', 'pm', 'reminder', 'jan', 'feb', 'march', 'apr', 'april', 'may', 'jun', 'jul',
                    'aug', 'sep', 'oct', 'nov', 'dec', 'spring', 'summer', 'autumn', 'winter', 'moved', 'none',
                    'doherty', 'january', 'february', 'march', 'may', 'june', 'july', 'august', 'september', 'october'
                                                                                                             'december'}

stop_words = stop_words.union(extra_stop_words)

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
        filtered_sentence = [w for w in word_tokens if w not in stop_words]

        results = {'Computer Science': -2, 'Politics': -2, 'Biology': -2, 'Chemistry': -2,
                   'Physics and Astronomy': -2,
                   'Languages': -2, 'Education': -2, 'Business': -2, 'Performing Arts': -2, 'Mathematics': -2,
                   'Other': 0, 'Engineering': -2, 'Art': -2}

        tagged = nltk.pos_tag(filtered_sentence)
        filtered_sentence = [key for key, value in tagged if value == 'NN' or value == 'NNS']
        print(nltk.pos_tag(filtered_sentence))

        for word in filtered_sentence:
            # print('Result for %s: ' % word)

            # Checks if the word is a keyword, if so then it's score gets a boost
            for k, v in data.items():
                if word in v:
                    current_val = results[k]
                    current_val += 10
                    results[k] = current_val

            # Use word embeddings to calculate relative score to other categories
            resultStream = process(word)
            if resultStream is not None:

                for key, value in resultStream.items():
                    current_value = results[key]
                    current_value += value
                    results[key] = current_value

        print("Results are as follows: ")
        c = Counter(results)
        most_likely = c.most_common(3)
        print([key for key, val in most_likely])
        print()
