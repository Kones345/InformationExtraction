
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from pathlib import Path
from collections import Counter
from regex_store import *


class Ontology:
    # Load the whole embedding matrix
    embeddings_index = {}
    categories = {}
    data_embeddings = {}

    def __init__(self):
        """
        Opens glove file and stores all the word embeddings in an array
        """
        with open('glove.6B.200d.txt') as f:
            for line in f:
                values = line.split()
                word = values[0]
                embed = np.array(values[1:], dtype=np.float32)
                self.embeddings_index[word] = embed
        # print('Loaded %s word vectors.' % len(self.embeddings_index))
        self.setup()

    # Fills the categories and data embeddings dictionaries
    def setup(self):
        # Words -> category, makes a dictionary of keyword; category pairs
        self.categories = {word: key for key, words in self.data.items() for word in words}

        # Embeddings for available words -  stores all known keywords embeddings
        self.data_embeddings = {key: value for key, value in self.embeddings_index.items() if
                                key in self.categories.keys()}

    # Stores all the keywords for different topica in the ontology
    data = {
        'Computer Science': ['software', 'object-oriented', 'architecture', 'design', 'product', 'debug', 'breakpoint',
                             'planning', 'a*', 'd*', 'development', 'quality', 'tests', 'artificial', 'intelligence',
                             'recursion',
                             'iterative', 'robots', 'logic'
                             'machine learning', 'ai', 'robotics', 'vision', 'data', 'complexity', 'search',
                             'nlp', 'natural language processing', 'ieee', 'navigation', 'robot', 'planning',
                             'humanoid', 'autonomous', 'knowledge', 'language', 'decision', 'recognition',
                             'classification', 'prediction', 'human', 'user', 'friendly', 'interaction', 'graphics',
                             'fun', 'usable', 'design', 'hci', 'interface', 'style', 'communication', 'visual',
                             'render', 'parallel', 'distributed', 'network', 'synchronization', 'efficient',
                             'synchronous', 'asynchronous', 'thread', 'multi', 'breach', 'cryptography', 'backdoor',
                             'encryption', 'decryption', 'hacking', 'algorithm', 'code', 'programming', 'java',
                             'python',
                             'ruby', 'robotics', 'circuit', 'computing', 'systems', 'framework', 'computer'],
        'Biology': ['bio', 'disease', 'immune', 'immunonoculation', 'biological', 'genome', 'biochemistry', 'molecules',
                    'medicine', 'clinic', 'cancer', 'health', 'cellular', 'cells', 'respiration', 'brain',
                    'neurological','enzyme', 'protein', 'glucose', 'fructose', 'biomolecule', 'biology',
                    'imaging', 'illness', 'sick', 'disease', 'healthcare'],
        'Chemistry': ['chemistry', 'drugs', 'polymers', 'graft', 'extruders', 'nanotechnology', 'fluids',
                      'thermodynamic','molecule', 'molecular',
                      'microemulsions', 'flourescence', 'water', 'dissolved', 'exothermic', 'endothermic', 'alcohol'],
        'Physics and Astronomy': ['physics', 'thermodynamics', 'nanotechnology', 'magnetism', 'frequency', 'nuclear',
                                  'stars',
                                  'cosmology', 'black', 'hole', 'resistance', 'voltage', 'current', 'mechanics',
                                  'momentum',
                                  'velocity',
                                  'acceleration', 'particles', 'electricity', 'fields', 'astrophysics', 'photon',
                                  'radiation', 'electromagnet', 'lepton', 'diffraction',
                                  'interference', 'motion',
                                  'projectile', 'gravity', 'telescope', 'Quasars', 'semiconductor', 'electronic',
                                  'circuit',
                                  'integrated', 'multimeter', 'voltage', 'controller'],
        'Politics': ['politics', 'social', 'public', 'policy', 'issue', 'environment', 'trend',
                     'economy', 'media', 'global', 'regulations', 'international', 'crisis',
                     'activist', 'prospects', 'welfare', 'community', 'movement', 'liberal', 'republican',
                     'democracy', 'communist', 'left', 'right', 'centre', 'totalitarian', 'dictatorship',
                     'vote', 'election', 'rally', 'ballot'],
        'Languages': ['language', 'phonology', 'english', 'writing', 'french', 'spanish', 'german', 'finnish',
                      'chinese', 'japanese', 'dictionary', 'lexicon', 'dialect', 'reading', 'listening',
                      'comprehension',
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
        'Education': ['education', 'academia', 'alumni', 'teaching', 'school', 'graduate', 'academic', 'grade',
                      'degree',
                      'university', 'college', 'school', 'nursery', 'homework', 'assignment', 'dissertation',
                      'doctorate',
                      'masters', 'phd', 'research'],
        'Engineering': ['material', 'engineering', 'environment', 'design', '3d', 'aeroacoustics', 'aerothermodynamics',
                        'air', 'turbulence', 'bridge', 'collapse', 'building', 'cad', 'car', 'fiber', 'optics',
                        'geoengineering', 'nano', 'nanotubes', 'stereo', 'tunnel', 'environmental', 'steel', 'metal'],
        'Mathematics': ['angle', 'measure', 'prove', 'solve', 'problem', 'equation', 'graph', 'plane', 'line', 'axis',
                        'algebra',
                        'adjacent', 'coefficient', 'frequency', 'circumference', 'denominator', 'distribution',
                        'polygon',
                        'expression', 'factorise', 'formula', 'frequency', 'density', 'gradient', 'hcf', 'indices',
                        'integer', 'quartile', 'rational', 'irrational', 'lcm', 'average', 'mean', 'median', 'mode',
                        'numerator', 'even', 'odd', 'parallel', 'perpendicular', 'probability', 'product', 'prime',
                        'quadratic', 'remainder', 'rotation', 'rotate', 'sum', 'symmetry', 'tangent', 'volume',
                        'solve', 'square', 'root', 'cube', 'mathematics', 'mathematical'],
        'Art': ['art', 'fine', 'color', 'colour', 'paint', 'gallery', 'print', 'pop', 'abstract', 'artist',
                'wall', 'acrylic', 'sculpture', 'watercolor', 'oil', 'supplies', 'modern', 'artwork', 'famous',
                'deco', 'pixel', '3d', 'liberal', 'vector', 'clip', 'creative', 'paper', 'nude', 'concept', 'animation'
                                                                                                            'crafts',
                'clay', 'tattoo', 'pencil', 'graphic', 'portfolio', 'decor'],
        'Medicine': ['surgery', 'diagnosis', 'doctor', 'nurse', 'heart', 'brain', 'medicine', 'hospital', 'gp', 'clinic'
                     'diagnose', 'pills', 'drugs', 'healthcare', 'coma', 'camcer', 'treatment', 'baby', 'steroids',
                     'x-ray', 'radiotheraphy', 'chemo', 'chemotherapy', 'radiation'],
        'Philosophy and Religion': ['theology', 'theological', 'religion', 'philosophy', 'jesus', 'christian',
                                    'christianity', 'bible', 'islam', 'allah', 'quran', 'mohamed', 'ramadan', 'eid',
                                    'buddha', 'hindu', 'hinduism', 'moral', 'morality', 'faith', 'atheist',
                                    'agnostic', 'sunni', 'shia', 'jew', 'judaism', 'torah', 'sabbath', 'holy',
                                    'sacred', 'spirit', 'pray', 'worship', 'church', 'pope', 'roman', 'catholic',
                                    'protestant', 'agnostic', 'saint', 'blessed', 'mosque', 'temple', 'ganesh',
                                    'god', 'plato', 'socrates', 'aristotle'],
        'Other': []

    }

    # Stop words to remove from the topic to improve results
    extra_stop_words = {'lecture', 'seminar', 'talk', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
                        'saturday', 'sunday', 'telecastseminar', 'today', 'weh', 'adamson', 'wing', 'hall', 'baker',
                        'rescheduled', 'am', 'pm', 'reminder', 'jan', 'feb', 'march', 'apr', 'april', 'may', 'jun',
                        'jul',
                        'aug', 'sep', 'oct', 'nov', 'dec', 'spring', 'summer', 'autumn', 'winter', 'moved', 'none',
                        'doherty', 'january', 'february', 'march', 'may', 'june', 'july', 'august', 'september',
                        'october'
                        'december'}

    def process(self, query):
        """
        Gives a word a score of how similar it is to each category
        :param query: the word to query
        :return: dictionary of scores for each category
        """
        scores = {}

        try:
            query_embed = self.embeddings_index[query]  # gets the embeddings for the word in question
        except:
            for c in self.categories:
                scores[c] = 0
            return None

        for word, embed in self.data_embeddings.items():  # iterates over the embeddings for known words
            category = self.categories[word]  # gets the category of keyword
            dist = query_embed.dot(embed)  # dot product of array of indexes for the current word and each keywords
            dist /= len(self.data[category])  # divides the dot product by the number of words in that category
            scores[category] = scores.get(category,
                                          0) + dist  # Adds this calculated score to the current score in results
        return scores

    def search_body_for_topic(self, body):
        """
        Counts the number of occurrences of keywords for each category in the body of an email
        :param body: the body text of an email
        :return: the scores from the body
        """
        res = {'Computer Science': 0, 'Politics': 0, 'Biology': 0, 'Chemistry': 0, 'Physics and Astronomy': 0,
               'Languages': 0, 'Education': 0, 'Business': 0, 'Performing Arts': 0, 'Mathematics': 0, 'Other': 0,
               'Engineering': 0, 'Art': 0, 'Medicine': 0, 'Philsophy and Religion': 0}

        for k, v in self.data.items():
            for word in v:
                if word in body:
                    score = res[k]
                    score += 1
                    res[k] = score
        return res

    def run(self, directory):
        """
        Classifies every given file
        :param directory: the directory where the files we want to classify can be found
        """
        stop_words = set(stopwords.words('english'))

        stop_words = stop_words.union(self.extra_stop_words)

        lemma = WordNetLemmatizer()
        topicRegex = re.compile(topic_regx_str)
        cleanTextRegex = re.compile(special_char_regx_str)

        pathlist = Path(directory).glob('**/*.txt')

        output = open('ontology_results.txt', 'w')

        for path in pathlist:
            p = str(path)
            with open(p, 'r', encoding='utf-8') as f:
                fileMatch = re.search('[0-9]*.txt', p)
                filename = fileMatch.group()
                content = f.read()
                topic = re.search(topicRegex, content)
                if topic is None:
                    continue
                topic = topic.group(1).lower()
                topic = re.sub(cleanTextRegex, '', topic)
                normalized = " ".join(lemma.lemmatize(word) for word in topic.split())
                word_tokens = word_tokenize(normalized)

                # Gives every category a reduced score to account for some miscellaneous words classifying under topics
                results = {'Computer Science': -2, 'Politics': -2, 'Biology': -2, 'Chemistry': -2,
                           'Physics and Astronomy': -2,
                           'Languages': -2, 'Education': -2, 'Business': -2, 'Performing Arts': -2, 'Mathematics': -2,
                           'Other': 0, 'Engineering': -2, 'Art': -2, 'Medicine': -2, 'Philosophy and Religion': -2}

                for word in word_tokens:
                    for k, v in self.data.items():
                        if word in v:
                            current_val = results[k]
                            current_val += 10
                            results[k] = current_val

                filtered_sentence = [w for w in word_tokens if w not in stop_words]

                tagged = nltk.pos_tag(filtered_sentence)
                filtered_sentence = [key for key, value in tagged if value == 'NN' or value == 'NNS']

                for word in filtered_sentence:

                    # Checks if the word is a keyword, if so then it's score gets a boost
                    # for k, v in self.data.items():
                    #     if word in v:
                    #         current_val = results[k]
                    #         current_val += 10
                    #         results[k] = current_val

                    # Use word embeddings to calculate relative score to other categories
                    resultStream = self.process(word)
                    if resultStream is not None:

                        for key, value in resultStream.items():
                            current_value = results[key]
                            current_value += value
                            results[key] = current_value

                # print("Results are as follows: ")
                c = Counter(results)

                top = c.most_common(1)[0]

                if top[0] == 'Other':
                    # Splits the text into header and body
                    try:
                        header, body = re.search(header_body_regx_str, content).groups()
                    except:
                        # print(p)
                        continue
                    new_res = self.search_body_for_topic(body)
                    c1 = Counter(new_res)
                    most_likely = c1.most_common(3)
                    output.write(filename + ': ' + str(most_likely[0][0]) + '\n')
                    # print([key for key, val in most_likely])
                else:
                    most_likely = c.most_common(3)
                    output.write(filename + ': ' + str(most_likely[0][0]) + '\n')
                    # print([key for key, val in most_likely])
        output.close()
        print('Ontology Classification ✅')
