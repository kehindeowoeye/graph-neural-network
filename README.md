# Graph-Neural-Network
**Dependencies**

Python 3.8.5. 

Pytorch 1.7.1. 

dgl 0.6.1.   

Numpy 1.19.4. 

Matplotlib 3.3.3. 



**Instructions on how to run scripts**

• First of all, due to compute power/resource constraint, I have only focused on node
class prediction. There are two variants of the node class prediction task however.
First there is a script (node_label_prediction.py) for training, validation and testing
without consideration for the text information at each node. The network was trained
over 100 epochs with the training loss, training accuracy, validation accuracy as well
as the test accuracy printed every 10 epochs.
To train, use: python node_label_prediction.py - -mode ‘Train’
To test, use: python node_label_prediction.py - -mode ‘Test’
Other arguments can be used to change the hidden dimension and number of labels
(see code/script for details)


• There is however another script where the text information at each node have been
included but due to resource constraint using this, I have used the validation dataset
which is much smaller than the training dataset. In addition, I have used only the first
word in the sentences. The key addition here with respect to the above script is that,
the texts attached are embedded and projected further into a lower dimensional state
which is then concatenated with the state of the convolution of the node features.
The new state is then used for downstream node label prediction task.
To train, use: python node_label_prediction_with_text.py - -mode ‘Train’
To test, use: python node_label_prediction_with_text.py - -mode ‘Test’
