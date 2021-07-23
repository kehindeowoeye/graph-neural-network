import numpy as np
import json
import torch
import pandas as pd
import dgl
import scipy.sparse as sp
import itertools
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
import argparse
import os

if not os.path.exists('saves1'):
    os.makedirs('saves1')


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
for i in test_data_graph['nodes']:
    test_data_node_id.append(i['id'])


#defines network object for test data
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
for i in train_data_graph['nodes']:
    train_data_node_id.append(i['id'])


#defines network object for training data
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
for i in valid_data_graph['nodes']:
    valid_data_node_id.append(i['id'])
    
#defines network object for validation data
valid_G = dgl.DGLGraph((torch.tensor(valid_data_source), torch.tensor(valid_data_target)))


class GCN(nn.Module):
    def __init__(self,in_feats,h_feats,num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g,h)
        return h
    




def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0
    

    
    #features = g.ndata['feat']
    features_train  =   torch.tensor(train_data_feature, dtype=torch.float32)
    features_valid  =   torch.tensor(valid_data_feature, dtype=torch.float32)
    features_test  =   torch.tensor(test_data_feature, dtype=torch.float32)
    train_labels =   np.argmax(torch.tensor(train_data_labels),1)
    test_labels  =   np.argmax(torch.tensor(test_data_labels),1)
    valid_labels =   np.argmax(torch.tensor(valid_data_labels),1)

    
    for e in range(100):
        
        #Forward
        logits_train = model(train_G,features_train)
        model_eval = model.eval()
        logits_valid = model_eval(valid_G,features_valid)
        logits_test  = model_eval(test_G,features_test)

        
        #Compute prediction
        pred_train = logits_train.argmax(1)
        pred_valid = logits_valid.argmax(1)
        pred_test  = logits_test.argmax(1)
       
        
        
        loss = F.cross_entropy(logits_train, train_labels)
        train_acc  = (pred_train == train_labels).float().mean()
        val_acc =  (pred_valid == valid_labels).float().mean()
        test_acc =  (pred_test ==  test_labels).float().mean()
       
        
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        if e % 10 == 0:
            print('In epoch {}, loss: {:.3f}, train_acc {:.3f} val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(e, loss, train_acc, val_acc, best_val_acc, test_acc, best_test_acc) )
            
            fn = 'saves1/graph_node_prediction_without_textstate_dict_'+str(e)+'b'+str(100)+'.pth'
            torch.save(model.state_dict(), fn)
            #print('Saved model to' + fn)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='node_prediction_without_text')
    parser.add_argument("-class", "--num_classes", default=121, type=int, help="number_of_labels")
    parser.add_argument("-h_dim", "--hidden", default=256, type=int, help="hidden_dimension")
    parser.add_argument("-mod", "--mode", default= 'Train', type=str, help="Train/Test")
    args = parser.parse_args()
    
    
    
    model = GCN(train_data_feature.shape[1], args.hidden, args.num_classes)
    
    if args.mode == 'Train':
        train(train_G, model)
    else:
        model.load_state_dict(torch.load('saves1/graph_node_prediction_without_textstate_dict_90b100.pth'))
        
        model.eval()
        features_test  =   torch.tensor(test_data_feature, dtype=torch.float32)
        test_labels  =   np.argmax(torch.tensor(test_data_labels),1)
        logits_test  = model(test_G,features_test)
        pred_test  = logits_test.argmax(1)
        test_acc =  (pred_test ==  test_labels).float().mean()
        print('test_accuracy is: {:.3f}'.format(test_acc))

        
    
