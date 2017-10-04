"""
9/18/17
kasim 
se.kasim.ebrahim@gmail.com
"""

import numpy as np
import string
import collections


# read data from file
def load():
    pos_data = []
    neg_data = []

    with open("data/rt-polarity.pos", 'r', encoding="latin-1") as pos_file:
        for line in pos_file:
            pos_data.append(line)

    with open("data/rt-polarity.neg", 'r', encoding="latin-1") as neg_file:
        for line in neg_file:
            neg_data.append(line)

    tot_data = pos_data + neg_data
    labels = [1] * len(pos_data) + [0] * len(neg_data)

    # print(len(labels), len(tot_data))

    return tot_data, labels


def normalize(list_text, target):
    # lower case
    _list = [text.lower() for text in list_text]
    # remove punctuation
    _list = [''.join(l for l in text if l not in string.punctuation) for text in _list]
    # trim whitespace
    _list = [' '.join(text.split()) for text in _list]

    # remove non informative lines i:e lines with only one word or empty lines
    _lebel = [target[ix] for ix, text in enumerate(_list) if len(text.split()) > 1]
    _list = [text for text in _list if len(text.split()) > 1]

    return _list, _lebel


"""
build word_dict--> a dictionary containing word and index the most frequent words comes first, the first word 
                    being RARE to represent infrequent words 
         count --> list of items each item contains word and its frequency
"""


def build_dictionary(list_text, vocabulary_size):
    words = [w for line in list_text for w in line.split()]

    count = [["RARE", -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))

    word_dict = {}
    for word, freq in count:
        word_dict[word] = len(word_dict)
    return word_dict, count


# prepares list of sentences with each sentence represented as a list of indices of words
def text2indices(sentences, word_dict):
    sentences_of_ind = []

    for sentence in sentences:
        sentence_ind = []
        for word in sentence.split():
            if word in word_dict:
                sentence_ind.append(word_dict[word])
            else:
                sentence_ind.append(0)
        sentences_of_ind.append(sentence_ind)
    return sentences_of_ind


def generate_batch(sentences_of_ind, batch_size, window_size):
    batch_items = []
    batch_neighbors = []
    # selected_sent = []
    for batch in range(batch_size):
        rand_sentence = np.random.choice(sentences_of_indexes)
        # selected_sent.append(rand_sentence)
        window_dict = [(x, rand_sentence[max(ix - window_size, 0):ix]+rand_sentence[ix+1:ix+window_size+1]) for ix, x in enumerate(rand_sentence)]
        batch_neighbors += [item[1] for item in window_dict]
        batch_items += [item[0] for item in window_dict]
    batch_dataset = [(x,y) for x,z in zip(batch_items, batch_neighbors) for y in z ]

    x = [i[0] for i in batch_dataset]
    y = [j[1] for j in batch_dataset]
    return x,y#, selected_sent


data, label = load()
# print(data[0], data[1])

_list, _label = normalize(data, label)
# print(_list[0], _label[0])

word_dict, count = build_dictionary(_list, 10000)
# print(count)
# print(word_dict)

sentences_of_indexes = text2indices(_list, word_dict)
# print(sentences_of_indexes)

# valid_words = ["cliche", "love", "hate", "silly", "sad"]
# print([word_dict[ind] for ind in valid_words])

batch_input, batch_output = generate_batch(sentences_of_indexes, 2, 2)
print(batch_input)
print(batch_output)
print([(x,y) for x,y in zip(batch_input,batch_output)])
# print(selected_sent)