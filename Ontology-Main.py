import spacy
import itertools
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

data = {
    'Computer Science':  ['software', 'object-oriented', 'architecture', 'design', 'product',
                           'development', 'quality', 'tests', 'artificial', 'intelligence', 
                           'machine learning', 'ai', 'robotics', 'vision',
                           'nlp', 'natural language processing', 'ieee', 'navigation', 'robot',
                            'humanoid', 'autonomous', 'knowledge', 'language', 'decision', 'recognition',
                            'classification', 'prediction', 'human', 'user', 'friendly', 'interaction', 'graphics', 
                            'fun', 'usable', 'design', 'hci', 'interface', 'style', 'communication', 'visual', 
                            'render', 'parallel', 'distributed', 'network', 'synchronization', 'efficient',
                            'synchronous', 'asynchronous', 'thread', 'multi', 'breach', 'cryptography', 'backdoor', 
                            'encryption', 'decryption', 'hacking'],
    'Biology': ['bio', 'disease', 'immune', 'immunomodulation', 'biological', 'genome', 'biochemistry', 'molecules',
                            'medicine', 'clinic', 'cancer', 'health'],
    'Chemistry': ['chemistry', 'drugs', 'polymers', 'graft', 'extruders', 'nanotechnology','fluids', 'thermodynamic', 
                'microemulsions', 'flourescence', 'water','dissolved'],
    'Electronics': ['semiconductor', 'electronic', 'circuit', 'integrated'],
    'Physics': ['physics', 'thermodynamics', 'nanotechnology', 'magnetism', 'frequency','nuclear'],
    'Politics': ['politics', 'social', 'public', 'policy', 'issue', 'environment', 'trend',
                'economy', 'media', 'global', 'regulations', 'international', 'crisis',
                'activist', 'prospects', 'welfare', 'community', 'movement'],
    'Languages': ['language', 'phonology', 'english', 'writing'],
    'Performing Arts': ['music', 'genre', 'ensemble', 'classical', 'theater'],
    'Business' : ['entrepreneurship', 'innovation', 'opportunities', 'customer',
                'finance', 'stock', 'exchange', 'money', 'investment', 'trader', 'bank', 'trend',
                'interest', 'business', 'businessman', 'businesswoman', 'market',  'need', 'global', 'success',
                'office', 'economy', 'management', 'equity', 'competition', 'revenue', 'profit'],
    'Education': ['education', 'academia', 'alumni', 'teaching', 'school', 'graduate']

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
  query_embed = embeddings_index[query]
  scores = {}
  for word, embed in data_embeddings.items():
    category = categories[word]
    dist = query_embed.dot(embed)
    dist /= len(data[category])
    scores[category] = scores.get(category, 0) + dist
  return scores


print(process('code'))


# # We’ll use SpaCy which includes pre-trained vectors for the most common words using the GloVe Common Crawl
# nlp = spacy.load("en")

# # Something reasonable for restaurants
# topic_labels = [
#   "Food & Drinks",
#   "Service",
#   "Location"
# ]
# # Chosen arbitrarily based on the first things that came into my head (or stomach, as the case may be).
# topic_keywords = [
#   "food drink burger soda pancakes wine delicious",
#   "service waiter friendly polite chef wait",
#   "location atmosphere bathroom parking sketchy"
# ]
# # Parsings strings rather than lists is both faster and more convenient when using Spacy

# # Use pipe to run this in parallel
# topic_docs = list(nlp.pipe(topic_keywords,
#   batch_size=10000,
#   n_threads=3))
# topic_vectors = np.array([doc.vector 
#   if doc.has_vector else spacy.vocab[0].vector
#   for doc in topic_docs])
# print("Vector for topic “%s”:" %topic_labels[0])
# print(topic_vectors[0])

# keywords = ['pizza was gross', 'hostess was rude','dangerous neighborhood']
# keyword_docs = list(nlp.pipe(keywords,
#   batch_size=10000,
#   n_threads=3))
# keyword_vectors = np.array([doc.vector
#   if doc.has_vector else spacy.vocab[0].vector
#   for doc in keyword_docs])
# print("Vector for keyword “%s”: " % keywords[0])
# print(keyword_vectors[0])

# # use numpy and scikit-learn vectorized implementations for performance
# simple_sim = cosine_similarity(keyword_vectors, topic_vectors)
# topic_idx = simple_sim.argmax(axis=1)
# print(simple_sim)

# for k, i in zip(keywords, topic_idx):
#   print("“%s” is about %s" %(k, topic_labels[i]))