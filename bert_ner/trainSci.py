import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from model_scibert import Net
#from model import Net
from data_load import NerDataset, pad, VOCAB, tokenizer, tag2idx, idx2tag
import os
import numpy as np
import argparse

def train(model, iterator, optimizer, criterion):
    model.train()
    for i, batch in enumerate(iterator):
        words, x, is_heads, tags, y, seqlens = batch
        _y = y # for monitoring
        optimizer.zero_grad()
        logits, y, _ = model(x, y) # logits: (N, T, VOCAB), y: (N, T)

        logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
        y = y.view(-1)  # (N*T,)

        loss = criterion(logits, y)
        loss.backward()

        optimizer.step()

        if i==0:
            print("=====sanity check======")
            print("words:", words[0])
            print("x:", x.cpu().numpy()[0][:seqlens[0]])
            print("tokens:", tokenizer.convert_ids_to_tokens(x.cpu().numpy()[0])[:seqlens[0]])
            print("is_heads:", is_heads[0])
            print("y:", _y.cpu().numpy()[0][:seqlens[0]])
            print("tags:", tags[0])
            print("seqlen:", seqlens[0])
            print("=======================")


        if i%10==0: # monitoring
            print(f"step: {i}, loss: {loss.item()}")

def eval(model, iterator, f):
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch

            _, _, y_hat = model(x, y)  # y_hat: (N, T)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    ## gets results and save
    with open("temp", 'w') as fout:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
            assert len(preds)==len(words.split())==len(tags.split())
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                fout.write(f"{w} {t} {p}\n")
            fout.write("\n")

    ## calc metric
    y_true =  np.array([tag2idx[line.split()[1]] for line in open("temp", 'r').read().splitlines() if len(line) > 0])
    y_pred =  np.array([tag2idx[line.split()[2]] for line in open("temp", 'r').read().splitlines() if len(line) > 0])

    num_proposed = len(y_pred[y_pred>1])
    num_correct = (np.logical_and(y_true==y_pred, y_true>1)).astype(np.int).sum()
    num_gold = len(y_true[y_true>1])

    print(f"num_proposed:{num_proposed}")
    print(f"num_correct:{num_correct}")
    print(f"num_gold:{num_gold}")
    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        if precision*recall==0:
            f1=1.0
        else:
            f1=0

    final = f + ".P%.2f_R%.2f_F%.2f" %(precision, recall, f1)
    with open(final, 'w') as fout:
        result = open("temp", "r").read()
        fout.write(f"{result}\n")

        fout.write(f"precision={precision}\n")
        fout.write(f"recall={recall}\n")
        fout.write(f"f1={f1}\n")

    os.remove("temp")

    print("precision=%.2f"%precision)
    print("recall=%.2f"%recall)
    print("f1=%.2f"%f1)
    return precision, recall, f1

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--top_rnns", dest="top_rnns", action="store_true")
    parser.add_argument("--logdir", type=str, default="./kp_pretrained_www/")
    parser.add_argument("--trainset", type=str, default="/home/cilab/LabMembers/YS/WWW/finetuning/train.txt")
    parser.add_argument("--validset", type=str, default="/home/cilab/LabMembers/YS/WWW/finetuning/valid.txt")
    parser.add_argument("--checkpoint", type=str, default=False)
    #parser.add_argument("--trainset", type=str, default="conll2003/train.txt")
    #parser.add_argument("--validset", type=str, default="conll2003/valid.txt")
    hp = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Net(hp.top_rnns, len(VOCAB), device, hp.finetuning).cuda()
    if hp.checkpoint:
        print("load check point of model...")
        checkpoint = torch.load(hp.checkpoint)
        print(checkpoint.keys())
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    #model = nn.DataParallel(model)

    train_dataset = NerDataset(hp.trainset)
    eval_dataset = NerDataset(hp.validset)

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)
    
    eval_iter = data.DataLoader(dataset=eval_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=pad)

    optimizer = optim.Adam(model.parameters(), lr = hp.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    #lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.1)
    for epoch in range(1, hp.n_epochs+1):
        
        #model.rnn.flatten_parameters()
        loss = train(model, train_iter, optimizer, criterion)

        print(f"=========eval at epoch={epoch}=========")
        if not os.path.exists(hp.logdir): os.makedirs(hp.logdir)
        fname = os.path.join(hp.logdir, str(epoch))
        
        #model.rnn.flatten_parameters()
        precision, recall, f1 = eval(model, eval_iter, fname)

        #torch.save(model.state_dict(), f"{fname}.pt")
        torch.save({
            'epoch':hp.n_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        },f"{fname}.pt")
        torch.save(model, "latest_model.pt")
        print(f"weights were saved to {fname}.pt")