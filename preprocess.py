import torch
import io
import os
import codecs
import ast
import sys


def load_vectors(fname):
    # Below function is copied from https://FastText.cc/docs
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    words = []
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
        words.append(tokens[0])
    return words, data

def get_data(data_path):
    print('[Get training data from .txt files...]')
    train_data_path = os.path.abspath(data_path)
    file_list = [file_name for file_name in os.listdir(train_data_path) if '.txt' in file_name]
    files = [train_data_path + '/' + file_name for file_name in file_list]
    training_data = []
    for file in files:
        data = load_data_from_txt(file)
        training_data.extend(data)
    return training_data


def load_data_from_txt(file_path):
    with codecs.open(file_path, "r", "utf-8") as f:
        lines = f.readlines()
        data = []
        for line in lines:
            line = line.strip('\n')
            sentence = ast.literal_eval(line)
            data.append(sentence)
    return data


def build_vocabulary(data_list, special_tokens=None):
    token2id = {}
    id2token = set()
    for tokens in data_list:
        for token in tokens:
            id2token.add(token)
    id2token = sorted(list(id2token))
    if special_tokens:
        for token in special_tokens:
            id2token.insert(0, token)
    for i, token in enumerate(id2token):
        token2id[token] = i
    return token2id, id2token

def build_char_vocabulary(special_tokens=None):
    token2id = {}
    id2token = {}
    idx = 0
    if special_tokens:
        for token in special_tokens:
            token2id[token] = idx
            id2token[idx] = token
            idx += 1

    # all cases of korean char
    for i in range(44032, 55204):
        token2id[chr(i)] = idx
        id2token[idx] = chr(i)
        idx += 1

    # all cases of alphabet + special symbols
    for i in range(32, 127):
        token2id[chr(i)] = idx
        id2token[idx] = chr(i)
        idx += 1

    # add Decimal case
    for i in range(10):
        token2id[str(i)] = idx
        id2token[idx] = str(i)
        idx += 1

    return token2id, id2token


def main():
    EMBEDDING = sys.argv[2]
    RAW_DATA_PATH = sys.argv[1]
    VOCAB_PATH = "data/vocabulary/"
    if EMBEDDING == 'glove':
        GLOVE_PATH = './data/glove.txt'
        PROCESSED_GLOVE_PATH = './data/glove.pt'
        # 0. Prepare pre trained word embeddings and training data.
        if not os.path.exists(RAW_DATA_PATH):
            print("There is not training data. Please check path of existing data.")
            exit(0)

        # 1. Convert text data of FastText to .pt file
        print("Converting a Gloved file .txt to .pt")
        glove_words, glove_data = load_vectors(GLOVE_PATH)
        print("Number of words in Glove: %d." % len(glove_data))
        torch.save(glove_data, PROCESSED_GLOVE_PATH)
        words = glove_words
    else:
        FAST_TEXT_PATH = "./data/cc.ko.300.vec"
        PROCESSED_FAST_TEXT_PATH = "./data/fasttext.pt"

        # 0. Prepare pre trained word embeddings and training data.
        if not os.path.exists(RAW_DATA_PATH):
            print("There is not training data. Please check path of existing data.")
            exit(0)
        if not os.path.exists(FAST_TEXT_PATH):
            os.system("wget -O ./data/cc.ko.300.vec.gz https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.vec.gz")
            os.system("gzip -d ./data/cc.ko.300.vec.gz")
            os.system("rm ./data/cc.ko.300.vec.gz")

        # 1. Convert text data of FastText to .pt file
        print("Converting a FastText file .txt to .pt")
        fast_text_words, fast_text_data = load_vectors(FAST_TEXT_PATH)
        print("Number of words in FastText: %d." % len(fast_text_data))
        torch.save(fast_text_data, PROCESSED_FAST_TEXT_PATH)
        words = fast_text_words

    # 2. Separate data and create vocabulary files.
    print("Separating the raw data and creating vocab...")

    data = get_data(os.path.abspath(RAW_DATA_PATH))
    sentences = data
    word2id, id2word = build_vocabulary(sentences + words, ['<EOS>', '<SOS>', '<PAD>'])
    char2id, id2char = build_char_vocabulary(['<PAD>', '<EOS>', '<SOS>', '<UNK>'])

    # vocab path check
    if not os.path.exists(VOCAB_PATH):
        os.mkdir(VOCAB_PATH)
    torch.save(word2id, os.path.abspath(VOCAB_PATH + '/word2id.pt'))
    torch.save(id2word, os.path.abspath(VOCAB_PATH + '/id2word.pt'))
    torch.save(char2id, os.path.abspath(VOCAB_PATH + '/char2id.pt'))
    torch.save(id2char, os.path.abspath(VOCAB_PATH + '/id2char.pt'))

    print("Done.")


if __name__ == '__main__':
    main()