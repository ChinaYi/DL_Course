import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils import data
import torchvision
import torchvision.transforms as transforms

import numpy as np
import random
import argparse

from model import Model
batch_size = 32
epochs = 16
lr = 1e-3
l2_decay = 0
log_interval = 100



train_full_dataset = torchvision.datasets.SVHN(root = 'svhn', split = 'train', download= True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.SVHN(root = 'svhn', split = 'test', download=True, transform=transforms.ToTensor())

def train_and_test(train_loader, test_loader, load = 'no'):
    model = Model()
    model = model.cuda()
    save_file = 'models/{}.pth'.format('q1.a')

    if load == 'no':
        model.train()
        
        loss_layer = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_decay)
            
            for iter,(imgs, labels) in enumerate(train_loader):
                
                
                imgs = imgs.cuda()
                labels = labels.cuda()
                
                features, re = model(imgs)
                loss = loss_layer(re, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if iter % log_interval == 0:
                    print('{}-{}: Loss {}, Batch Accuracy {}'.format(epoch, iter, loss.item(), re.data.max(1)[1].eq(labels.data).sum()))
                    
        print('Training Done!')
        torch.save(model.state_dict(), save_file)
    else:
        model.load_state_dict(torch.load(save_file))
    # test
    
    acc_all = 0
    acc_classes = {k:0 for k in range(10)}
    cnt_classes = {k:0 for k in range(10)}
    with torch.no_grad():
        model.eval()
        
        for (imgs, labels) in test_loader:
            labels_np = labels.numpy()
            
            imgs = imgs.cuda()
            labels = labels.cuda()
            
            features, re = model(imgs)
            answer = re.max(1)[1].eq(labels).to('cpu').numpy()
            
            for index in range(len(answer)):
                cnt_classes[labels_np[index]] += 1
                if answer[index] == 1:
                    acc_classes[labels_np[index]] += 1
                    acc_all += 1
    
    print('Test Done! Acc is {}'.format(acc_all / test_dataset.__len__()))
    for k in range(10):
        print('Class {}, acc {}'.format(k, acc_classes[k] / cnt_classes[k]))
    
if __name__ == '__main__':
    train_loader = data.DataLoader(dataset = train_full_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle= True)
    train_and_test(train_loader, test_loader)
    
    