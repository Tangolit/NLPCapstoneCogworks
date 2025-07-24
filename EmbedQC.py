#1: Process captions/queries by lowercasing the text, 
# removing punctuation, and 
# tokenizing words based on white space. Refer to the “bag of words” exercise notebook for efficient code for striping punctuation out of a string

import re, string
from collections import Counter
import numpy as np
import gensim
from cogworks_data.language import get_data_path
from gensim.models.keyedvectors import KeyedVectors

filename = "glove.6B.200d.txt.w2v"

punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))

def strip_punc(corpus):
    """
    Removes all punctuation from string
    input: corpus (str) --> output: corpus with punc removed
    """
    return punc_regex.sub('', corpus)

def process_doc(doc):
    ''' 
    Converts input to all lowercase, removes punctuation, and splits each word by space
    input: doc (str) -> returns (list)
    '''
    return strip_punc(doc).lower().split()

#2: Take our vocabulary to be all words across all captions in the COCO dataset. 
# Treating each caption as its own “document” compute the inverse document frequency for each word in the vocabulary. 
# Efficiency is important here!

def to_counter(doc):
    """
    Produces word count of document
    """
    return Counter(doc)

def to_vocab(counters):
    """
    counters: Iterable[Iterable[str]]
    """
    vocab = Counter()
    for counter in counters:
        vocab.update(counter)
    return vocab

#annotations = coco_data["annotations"]
annotations = [{"hi": "hello", "test": "test2", "caption": "to see if this works"}, {"hi": "hello", "test": "test2", "caption": "to see if this works number"}] #replace w/ real data
all_captions = [annotation["caption"] for annotation in annotations]
print(all_captions)

def compute_idf(word, counters):
    """
    Computes inverse document frequency (IDF) for each term in doc 
    """
    N = len(counters)
    nt = sum(1 if word in counter else 0 for counter in counters)
    return np.log10(N / nt)

#tests:
captions_counters = [to_counter(process_doc(caption)) for caption in all_captions]
vocab = to_vocab(captions_counters)
idf = compute_idf("number", captions_counters)
print(idf)

#3: Make a function that can embed any caption/query text (using GloVe-200 embeddings weighted by IDFs of words across captions). An individual word not in the GloVe or IDF vocabulary should yield an embedding vector of just zeros.
path = get_data_path(filename)
glove = KeyedVectors.load_word2vec_format(path, binary = False)

print(glove["notaword"] == None)

def embed_text(text, vocabulary, counters):
    sum = np.zeros(200)
    word_list = process_doc(text)
    for word in word_list:
        if word not in vocabulary or word not in glove:
            return np.zeros(200)
        sum += (glove[word] * compute_idf(word, counters))
        print(sum)
    magnitude = np.linalg.norm(sum)
    print("magnitude: " + str(magnitude))
    
    unit_sum = sum / magnitude
    return unit_sum
    
for caption in all_captions:
    embedded_text = embed_text(caption, vocab, captions_counters)
    print(embedded_text)