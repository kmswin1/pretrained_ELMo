import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import time, math, json, os, sys
from args import get_train_args, get_test_args
from dataset import ELMoDataset
from model import ELMo

def main():
    opts = get_test_args()
    print ("load data ...")
    valid_data = ELMoDataset(opts, split='validation')
    validloader = DataLoader(valid_data, shuffle=True, batch_size=opts.batch_size)
    print ("load model ...")
    model = ELMo(opts, [valid_data.word_vocab_size, valid_data.char_size])
    model.load_state_dict(torch.load('model.pt'))
    loss = torch.nn.CrossEntropyLoss()
    model.cuda()

    model.eval()
    valid_loss = 0
    print ("Evaluating ...")
    tot = 0
    with torch.no_grad():
        for i, batch_data in enumerate(validloader):
            word_idx, char_idx = batch_data
            pred = model(word_idx, char_idx)
            batch_loss = loss(pred, word_idx[:,1:].reshape(-1))
            valid_loss += torch.sum(batch_loss)
            tot += word_idx.size(0)

    print ('valid loss : {} preplexity : {}'.format(valid_loss/tot, 2**(valid_loss/tot)))
if __name__ == '__main__':
    main()