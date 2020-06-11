from torch.utils.data import Dataset
import torch
import os
import ast

class ELMoDataset(Dataset):
    def __init__(self, opts, split='training'):
        self.opts = opts
        self.split = split
        data = self.get_data('data/dataset')
        self.sentences, self.max_len = self.build_padded_data(data)
        if os.path.exists('data/vocabulary/word2id.pt'):
            self.word2id = torch.load('data/vocabulary/word2id.pt')
            self.id2word = torch.load('data/vocabulary/id2word.pt')
            self.char2id = torch.load('data/vocabulary/char2id.pt')
            self.id2char = torch.load('data/vocabulary/id2char.pt')
        else:
            print("[ERROR: Run preprocess.py before training process]")
            exit(0)
        self.data_size = len(self.sentences)
        self.word_vocab_size = len(self.id2word)
        self.char_size = len(self.id2char)
        self.max_char_len = 62

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        word_idx = []
        char_idx = []
        for word in self.sentences[idx]:
            try:
                word_idx.append(self.word2id[word])
            except:
                word_idx.append(self.word2id['<UNK>'])
            temp = []
            if word == '<SOS>' or word == '<PAD>' or len(word) > 62:
                for _ in range(self.max_char_len):
                    temp.append(self.char2id['<PAD>'])
                temp.insert(0, self.char2id['<SOS>'])
                temp.append(self.char2id['<EOS>'])
                char_idx.append(temp)
                continue

            diff = self.max_char_len - len(word)
            for char in word:
                try:
                    temp.append(self.char2id[char])
                except:
                    temp.append(self.char2id['<UNK>'])
            temp.insert(0, self.char2id['<SOS>'])
            temp.append(self.char2id['<EOS>'])
            for _ in range(diff):
                temp.append(self.char2id['<PAD>'])
            char_idx.append(temp)
        word_idx = torch.tensor(word_idx, dtype=torch.long, device=self.opts.device)
        char_idx = torch.tensor(char_idx, dtype=torch.long, device=self.opts.device)
        return word_idx, char_idx

    def get_data(self, data_path):
        print('[Get {} data from .txt files...]'.format(self.split))
        data_path = os.path.abspath(data_path)
        file_list = [file_name for file_name in os.listdir(data_path) if self.split in file_name]
        files = [data_path + '/' + file_name for file_name in file_list]
        training_data = []
        for file in files:
            data = self.load_data_from_txt(file)
            training_data.extend(data)
        return training_data

    def load_data_from_txt(self, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
            data = []
            for line in lines:
                line = line.strip('\n')
                sentence = ast.literal_eval(line)
                # add <SOS> and <EOS> tokens
                sentence.insert(0, '<SOS>')
                sentence.append('<EOS>')
                data.append(sentence)
        return data

    def find_max_len(self, sentences):
        lens = [len(sentence) for sentence in sentences]
        max_len = max(lens)
        return max_len

    def build_padded_data(self, data):
        sentences = data
        max_len = 52
        for i, sentence in enumerate(sentences):
            diff = max_len - len(sentence)
            # add padding tokens
            for _ in range(diff):
                sentence.append('<PAD>')
        return sentences, max_len