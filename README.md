# The Impact of Context Span on Emotion Classification in Conversations

#### Team Members Information - Aditya Raj (22BDS002), Akash Nayak (22BDS003), Niharika R (22BDS050)

#### IMPORTANT: Please setup Hadoop (optional) for distributed file system management as the embedding files can get very big <br>
#### Download the src folder and download all the data in the src/data folder<br>
#### Run the script src/code/node2vec_graph_creation_script.py for generating the speaker graph.

#### Project Report Link - [https://docs.google.com/document/d/1eKzupUwlNBVfo8zvZM7SBaJVt7C00i4ZdpvMbNbRuJ0/edit](https://docs.google.com/document/d/18nTFf80qHTaip3tbqqrWkevjoprEKCucx80OEa2rxX0/edit?usp=sharing) <br>


## Models and algorithms utilised - <br>
1. RoBERTa
2. COMET - Commonsense Transformers for Automatic Knowledge Graph Construction
3. node2vec

Code for generating the embeddings of RoBERTa, COMET and ndoe2vec are put under their respective markdown cell.

Ensure you run the MELD Dataset Loader, to get the data loaded in correctly.

To make things easier, we have compiled all the embeddings and loaded all the data in a pickle file which can be imported as a dataframe. This will save both time and heavy computation. If required to run on a different dataset, please create a custom data loader which will match the input format of the respective dataset.

## Embedding Sizes Generated and Utilised for training - <br>
1. RoBERTa - 768
2. COMET - 768
3. node2vec - 256 (Customisable in the node2vec embedding generator.

