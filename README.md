# PyTorch-NLP


### Datasets 

| Dataset | Classes | Train Samples | Test Samples | 
| :--- | :---: | :---: | :---: | 
| AG's News | 4 | 120,000 | 7,600 | 
| Yelp Review Polarity | 2 | 560,000 | 38,000 |
| Yelp Review Full | 5 | 650,000 | 50,000 | 
| Yahoo! Answers | 10 | 1,400,000 | 60,000 | 

### Results

| Dataset | Logistic (10 epoch) | EmbeddingBag (10 epoch) | LSTM (10 epoch) | Pretrained TransformerEncoder (10 epoch) | Vanilla TransformerEncoder (2Layer)
| :--- | :---: | :---: | :---: | :---: | 
| AG's News | 99.37 | 96.48 | 98.07 | 95.85 | 99.39
| Yelp Review Polarity  | 98.10 | 96.58 | 96.72| 98.74 | 98.42
| Yelp Review Full  | 74.07 | 71.47 | 77.81 | 71.29| 82.8
| Yahoo! Answers | 77.56 | 84.1 | 64.35 | 77.15 | 74.12 

### Experimental settings

#### Logistic

ngram TF-IDF: min-ngram 1, max ngram 2, max features 50000


#### Embedding bag

Embed size 30


#### LSTM

Hidden size 256
num_layers 2


#### Pretrained Transformer

truncate 256



#### Custom Transformer

We also leverage some parts of the Huggingface BERT encoder to explore the impact of following factors:
 - Learnable position encoding vs fixed Cosine position encoding.
 - CLS embedding vs pooled embedding as the input text representation.

| Model Setting | AG's News | 
| :--- | :---: |  
| CLS + Leanable | 99.28 | 
| CLS + Cosine  | 95.13 |
| Pooled + Learnable | 99.38 | 

