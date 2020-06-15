import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import time, math, json, os, sys
from args import get_train_args
from dataset import ELMoDataset
from model import ELMo
from utils import EarlyStopping

def main():
    opts = get_train_args()
    print ("load data ...")
    data = ELMoDataset(opts)
    dataloader = DataLoader(data, shuffle=True, batch_size=opts.batch_size)
    valid_data = ELMoDataset(opts, split='validation')
    validloader = DataLoader(valid_data, shuffle=True, batch_size=opts.batch_size)
    print ("load model ...")
    model = ELMo(opts, [data.word_vocab_size, data.char_size])
    optimizer = optim.Adam(model.parameters(), lr=opts.learning_rate)
    early_stopping = EarlyStopping(5, 0.0)
    if opts.multi == True:
        model = torch.nn.DataParallel(model)
    if opts.resume == True:
        print ("resume training")
        model.load_state_dict(torch.load('model.pt'))
    model.cuda()
    loss = torch.nn.CrossEntropyLoss()
    train_batch_num = math.ceil(data.data_size / opts.batch_size)
    valid_batch_num = math.ceil(valid_data.data_size / opts.batch_size)

    print ("start training")
    for epoch in range(1, opts.epochs + 1):
        print ("epoch : " + str(epoch))
        model.train()
        epoch_start = time.time()
        epoch_loss = 0
        tot = 0
        for i, batch_data in enumerate(dataloader):
            optimizer.zero_grad()
            word_idx, char_idx = batch_data
            pred = model(word_idx, char_idx)
            train_loss = loss(pred, word_idx[:,1:].reshape(-1))
            train_loss.backward()
            optimizer.step()
            batch_loss = train_loss.item()
            tot += word_idx.size(0)
            print ('\r{:>10} epoch {} progress {} loss: {} perplexity : {}\n'.format('', epoch, tot/data.__len__(), batch_loss, 2**batch_loss), end='')
            epoch_loss += batch_loss
        end = time.time()
        time_used = end - epoch_start
        print ('one epoch time: {} minutes'.format(time_used/60))
        print ('{} epochs'.format(epoch))
        print ('epoch {} loss : {} perplexity : {}'.format(epoch, epoch_loss/train_batch_num, 2**(epoch_loss/train_batch_num)))

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for i, batch_data in enumerate(validloader):
                word_idx, char_idx = batch_data
                pred = model(word_idx, char_idx)
                batch_loss = loss(pred, word_idx[:,1:].reshape(-1))
                valid_loss += batch_loss.item()

        print ('valid loss : {} preplexity : {}'.format(valid_loss/valid_batch_num, 2**(valid_loss/valid_batch_num)))

        with open('log.txt', 'a') as f:
            f.write(str(epoch) + ' epoch :' + str(epoch_loss/train_batch_num) + ' ' + str(2**(epoch_loss/train_batch_num)) + '\n')
            f.write(str(epoch) + ' valid :' + str(valid_loss/valid_batch_num) + ' ' + str(2**(valid_loss/valid_batch_num)) + '\n')

        # check early stopping
        if early_stopping(valid_loss):
            print("[Training is early stopped in %d Epoch.]" % epoch)
            if not os.path.exists(opts.model_path):
                os.mkdir(opts.model_path)
            state_dict = model.state_dict()
            torch.save(state_dict, os.path.abspath(opts.model_path + '/model.pt'))
            print("[Saved the trained model successfully.]")
            break

        if epoch % opts.save_step == 0:
            print ("save model...")
            torch.save(model.state_dict(), 'model.pt')

if __name__ == '__main__':
    main()