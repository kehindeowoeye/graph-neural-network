import numpy as np
import json
import torch
import pandas as pd
import dgl
import scipy.sparse as sp
import itertools
import torch.nn as nn
import torch.nn.functional as F
import re
import unicodedata
import matplotlib.pyplot as plt
import argparse
import os
from dgl.nn import GraphConv

if not os.path.exists('saves2'):
    os.makedirs('saves2')


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s



#Loads and extract the test data
test_data_feature = np.load('test_feats.npy')
test_data_id = np.load('test_graph_id.npy')
test_data_labels = np.load('test_labels.npy')
with open('test_graph.json') as f:
  test_data_graph = json.load(f)

test_data_target = []
test_data_source = []
for i in test_data_graph['links']:
    test_data_target.append(i['target'])
    test_data_source.append(i['source'])

test_data_node_id = []
test_text = []
for i in test_data_graph['nodes']:
    test_text.append(normalizeString(i['text']).split()[0:1] )#selects just the  first word in sentence
    test_data_node_id.append(i['id'])

    
test_G = dgl.DGLGraph((torch.tensor(test_data_source), torch.tensor(test_data_target)))
    
    
    
#Loads and extract the train data
train_data_feature = np.load('train_feats.npy')
train_data_id = np.load('train_graph_id.npy')
train_data_labels = np.load('train_labels.npy')
with open('train_graph.json') as f:
  train_data_graph = json.load(f)

train_data_target = []
train_data_source = []
for i in train_data_graph['links']:
    train_data_target.append(i['target'])
    train_data_source.append(i['source'])
    
train_data_node_id = []
train_text = []
for i in train_data_graph['nodes']:
    train_text.append(normalizeString(i['text']).split()[0:1] )
    train_data_node_id.append(i['id'])


train_G = dgl.DGLGraph((torch.tensor(train_data_source), torch.tensor(train_data_target)))
    
    
    
       
#Loads and extract the validation data
valid_data_feature = np.load('valid_feats.npy')
valid_data_id = np.load('valid_graph_id.npy')
valid_data_labels = np.load('valid_labels.npy')
with open('valid_graph.json') as f:
  valid_data_graph = json.load(f)

valid_data_target = []
valid_data_source = []
for i in valid_data_graph['links']:
    valid_data_target.append(i['target'])
    valid_data_source.append(i['source'])
    
valid_data_node_id = []
val_text = []
for i in valid_data_graph['nodes']:
    val_text.append(normalizeString(i['text']).split()[0:1] )
    valid_data_node_id.append(i['id'])
    

valid_G = dgl.DGLGraph((torch.tensor(valid_data_source), torch.tensor(valid_data_target)))



class GCN(nn.Module):
    #graph module
    def __init__(self,in_feats,h_feats,num_classes,out_embed):
        super(GCN, self).__init__()
        
        self.h_feats = h_feats
        self.out_embed = out_embed
        self.num_classes = num_classes
        self.in_feats = in_feats
        
        
        self.conv1 = GraphConv(self.in_feats, self.h_feats)
        self.conv2 = nn.Linear(self.out_embed,2)
        self.conv4 = GraphConv(self.h_feats+2, self.num_classes)
        
        
        self.embed = nn.Sequential(
                 nn.Embedding(self.in_feats, self.h_feats),
                 nn.Flatten(start_dim=1))
        
        
    def forward(self, g, in_feat, encoder_input_data):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        embedding_output = self.embed(encoder_input_data)
        embedding_output  = self.conv2(embedding_output)
        h = torch.cat((h , embedding_output), dim = 1)
        h = self.conv4(g,h)
        return h
        
        
def process_txt(val_text,test_text):
    #converts texts for downstream tasks including
    #padding, tokenization among others
    input_word = set()
    for sentence in val_text:
        for word in sentence:
            if word not in input_word:
                input_word.add(word)

    input_word = sorted(list(input_word))
    num_encoder_tokens = len(input_word)
    max_encoder_seq_length = max([len(txt) for txt in val_text])




    input_token_index = dict(
     [(char, i) for i, char in enumerate(input_word)])


    encoder_input_data = torch.zeros(
      (len(val_text), max_encoder_seq_length, num_encoder_tokens),dtype=torch.long);#print(encoder_input_data.shape)
    encoder_test_data =  torch.zeros(
    (len(test_text), max_encoder_seq_length, num_encoder_tokens),dtype=torch.long)


    for i, val_txt_ in enumerate(val_text):
        for t, word in enumerate(val_txt_):
            encoder_input_data[i, t, input_token_index[word]] = 1
            
    test_token_index = {}
    for i, tst_txt_ in enumerate(test_text):
        for t, word in enumerate(tst_txt_):
            if word  not in input_word:
                test_token_index[word] = len(input_word)+1
            else:
                test_token_index[word] =  input_token_index[word]
            
    for i, tst_txt_ in enumerate(test_text):
        for t, word in enumerate(tst_txt_):
            encoder_test_data[i, t, test_token_index[word]] = 1
    
            
    return encoder_input_data, encoder_test_data
            


def train(g, model, encoder_input_data,test_input_data):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0;best_test_acc = 0
    
    #converts node features to a torch sensor
    features_train  = torch.tensor(train_data_feature, dtype=torch.float32)
    features_valid  = torch.tensor(valid_data_feature, dtype=torch.float32)
    features_test  =  torch.tensor(test_data_feature,  dtype=torch.float32)
    
    
    #gets the labels for the one hot encoding
    train_labels =    np.argmax(torch.tensor(train_data_labels),1)
    test_labels  =    np.argmax(torch.tensor(test_data_labels),1)
    valid_labels =    np.argmax(torch.tensor(valid_data_labels),1)
    
   

        

    
    for e in range(60):
        
        #Forward
        logits_valid = model(valid_G,features_valid,encoder_input_data)
        model_eval = model.eval()# evaluation/testing mode
        logits_test  = model_eval(test_G,features_test,test_input_data)
        
        
        #Compute prediction
        pred_valid = logits_valid.argmax(1)
        pred_test  = logits_test.argmax(1)
        
        

        #loss and comparison with ground truth
        loss = F.cross_entropy(logits_valid, valid_labels)
        val_acc =   (pred_valid == valid_labels).float().mean()
        test_acc =  (pred_test  ==  test_labels).float().mean()
   
       
        
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        if e % 10 == 0:
            print('In epoch {}, loss: {:.3f},  val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(e, loss, val_acc, best_val_acc, test_acc, best_test_acc) )
            fn = 'saves2/graph_state_dict_'+str(e)+'b'+str(100)+'.pth'
            torch.save(model.state_dict(), fn)
           
            
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='node_prediction_without_text')
    parser.add_argument("-class", "--num_classes", default=121, type=int, help="number_of_labels")
    parser.add_argument("-h_dim", "--hidden", default=256, type=int, help="hidden_dimension")
    parser.add_argument("-mod", "--mode", default= 'Train', type=str, help="Train/Test")
    args = parser.parse_args()
    
    val_text_data, test_text_data = process_txt(val_text,test_text)
    model = GCN(train_data_feature.shape[1], args.hidden, args.num_classes, val_text_data.shape[2]*args.hidden)
    
    
    if args.mode == 'Train':
        train(train_G, model,val_text_data,test_text_data )
    else:
        model.load_state_dict(torch.load('saves2/graph_state_dict_50b100.pth'))
        model.eval()
        features_test  =  torch.tensor(test_data_feature,  dtype=torch.float32)
        test_labels  =    np.argmax(torch.tensor(test_data_labels),1)
        logits_test  = model(test_G,features_test, test_text_data )
        pred_test  = logits_test.argmax(1)
        test_acc =  (pred_test ==  test_labels).float().mean()
        print('test_accuracy is: {:.3f}'.format(test_acc))
