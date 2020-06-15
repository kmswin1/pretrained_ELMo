import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import time, math, json, os, sys
from args import get_train_args, get_test_args
from dataset import ELMoDataset
from model import ELMo

PWD = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/hanmail/projects/dha/dha_python_wrapper-2.7.1/install/lib')
import libpydha as pydha

class DataConverter():
    def __init__(self, dha, data_path):
        self.data_path = data_path
        self.word2id = torch.load(os.path.abspath(data_path + '/vocabulary/word2id.pt'))
        self.id2word = torch.load(os.path.abspath(data_path + '/vocabulary/id2word.pt'))
        self.char2id = torch.load(os.path.abspath(data_path + '/vocabulary/char2id.pt'))
        self.id2char = torch.load(os.path.abspath(data_path + '/vocabulary/id2char.pt'))
        self.word_vocab_size = len(self.id2word)
        self.char_vocab_size = len(self.id2char)
        self.analyzer = 'hanl'
        self.option = 'morph,char_pos,m_tag,verb_input,ncp_input,nbu_cat,nbu_num,suffix_cat,prefix_cat,josa_cat,ns_suffix_cat,cj1,level_9'
        self.tokenizer = dha

    def get_max_len(self):
        return max(list(map(lambda x : len(x), self.id2word)))

    def get_id_sequence(self, sentence):
        word_idx = []
        pos_idx = []
        char_idx = []
        tokens = self.tokenizer.analyze(sentence, self.analyzer, self.option)
        words, pos, cpos, clen = self.remove_duplicate_tokens(sentence, json.loads(tokens))
        max_len = 62
        for word in words:
            try:
                word_idx.append(self.word2id[word])
            except:
                word_idx.append(self.word2id["<UNK>"])
        words.insert(0, '<SOS>')
        words.append('<EOS>')
        for word in words:
            cur_word = word
            temp = []
            if cur_word == '<SOS>' or cur_word == '<EOS>' or cur_word == '<PAD>':
                for _ in range(max_len):
                    temp.append(self.char2id['<PAD>'])
                temp.insert(0, self.char2id['<SOS>'])
                temp.append(self.char2id['<EOS>'])
                char_idx.append(temp)
                continue

            diff = max_len - len(cur_word)
            for char in cur_word:
                try:
                    temp.append(self.char2id[char])
                except:
                    temp.append(self.char2id['<UNK>'])
            temp.insert(0, self.char2id['<SOS>'])
            temp.append(self.char2id['<EOS>'])
            for _ in range(diff):
                temp.append(self.char2id['<PAD>'])
            char_idx.append(temp)
        word_idx.insert(0, self.word2id['<SOS>'])
        word_idx.append(self.word2id["<EOS>"])
        word_tensor = torch.tensor(word_idx, dtype=torch.long).unsqueeze(0)
        char_tensor = torch.tensor(char_idx, dtype=torch.long).unsqueeze(0)
        del words[len(words)-1]
        del words[0]
        return word_tensor, char_tensor, words, pos, cpos, clen

    def remove_duplicate_tokens(self, sentence, tokens):
        position = 0
        words = []
        pos = []
        cpos = []
        clen = []
        selected_tokens = []
        for token in tokens:
            if token['cpos'] >= position and token['clen'] > 0:
                position = token['cpos'] + token['clen']
                selected_tokens.append(token)
            elif token['cpos'] < position and token['cpos'] + token['clen'] > position:
                selected_tokens = [t for t in selected_tokens if t['cpos'] < token['cpos']]
                selected_tokens.append(token)
                position = token['cpos'] + token['clen']
        for token in selected_tokens:
                words.append(token['str'])
                pos.append(token['tag'])
                cpos.append(token['cpos'])
                clen.append(token['clen'])
        return words, pos, cpos, clen

    def get_word_label(self, labels):
        result = []
        for label in labels[0]:
            result.append(self.id2ner[label])
        return result

    def get_tagged_sentence(self, words, label):
        label = self.get_word_label(label)
        result = []
        previous = 'O'
        tag = ''
        for i, word in enumerate(words):
            if previous != 'O' and (label[i].startswith('B') or label[i] == 'O'):
                result.append("<" + tag + ">")
            if label[i].startswith('B'):
                tag = label[i]
            result.append(word)
            previous = label[i]
        return ''.join(result)

def main():
    opts = get_test_args()
    print ("load data ...")
    # DHA config
    coll_name = 'default'
    dic_path = '/hanmail/projects/dha/dha_resources-2.9.41-default/'
    dha = pydha.Index()
    dha.init(dic_path, coll_name)

    converter = DataConverter(dha, 'data')
    print ("load model ...")
    model = ELMo(opts, [converter.word_vocab_size, converter.char_vocab_size])
    model.load_state_dict(torch.load('model.pt', map_location='cpu'))
    loss = torch.nn.CrossEntropyLoss()

    model.eval()
    print ("Evaluating ...")
    with torch.no_grad():
        while True:
            sentence = input()
            word_idx, char_idx, parsed_word, parsed_pos, cpos, clen = converter.get_id_sequence(sentence)
            pred = model(word_idx, char_idx)
            print (converter.id2word[torch.argmax(pred[-1])])

if __name__ == '__main__':

    main()