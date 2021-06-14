import numpy as np

import nltk
nltk.download('stopwords')

def create_label_index_maps(labels):
    label_to_index = {}
    index_to_label = {}
    for i, label in enumerate(labels):
        label_to_index[label] = i
        index_to_label[i] = label
    return label_to_index, index_to_label


def make_one_hot(y, label_to_index):
    labels = list(label_to_index.keys())
    n_classes = len(labels)
    y_new = []
    for label in y:
        current = np.zeros(n_classes)
        i = label_to_index[label]
        current[i] = 1.0
        y_new.append(current)
    y_new = np.asarray(y_new)
    return y_new


def calculate_df_doc_freq(df):
    docfreq = {}
    docfreq["UNK"] = len(df)
    for index, row in enumerate(df):
        line = row
        words = line.strip().split()
        temp_set = set(words)
        for w in temp_set:
            try:
                docfreq[w] += 1
            except:
                docfreq[w] = 1
    return docfreq

def calculate_doc_freq(docs):
    docfreq = {}
    for doc in docs:
        words = doc.strip().split()
        temp_set = set(words)
        for w in temp_set:
            try:
                docfreq[w] += 1
            except:
                docfreq[w] = 1
    return docfreq


def calculate_inv_doc_freq(df, docfreq):
    inv_docfreq = {}
    N = len(df)
    for word in docfreq:
        inv_docfreq[word] = np.log(N / docfreq[word])
    return inv_docfreq



def create_word_index_maps(word_vec):
    word_to_index = {}
    index_to_word = {}
    words = word_vec
    for i, word in enumerate(words):
        word_to_index[word] = i
        index_to_word[i] = word
    return word_to_index, index_to_word


def preprocess(df):
    print("Preprocessing data for Tf-Idf..")
    new_words = set()

    for index, row in enumerate(df):
        if index % 10000 == 0:
            print("Finished rows: " + str(index) + " out of " + str(len(df)))
        line = row
        words = line.strip().split()

        for word in words:
            new_words.add(word)
    return list(new_words)

def get_label_docs_dict(df, label_term_dict, pred_labels):
    label_docs_dict = {}
    for l in label_term_dict:
        label_docs_dict[l] = []
    for index, row in enumerate(df):
        line = row
        label_docs_dict[pred_labels[index]].append(line)
    return label_docs_dict


def print_label_term_dict(label_term_dict, components, print_components=True):
    for label in label_term_dict:
        print(label)
        print("*" * 80)
        for val in label_term_dict[label]:
            try:
                if print_components:
                    print(val, components[label][val])
                else:
                    print(val)
            except Exception as e:
                print("Exception occurred: ", e, val)
