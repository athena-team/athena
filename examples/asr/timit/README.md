
#### Data Resource
The dataset can be downloaded from https://github.com/philipperemy/timit.


#### Experimental Details
The training config we used is mtl_transformer_sp.json. Models are trained with 48 phonemes, and results are mapped to 39 phonemes for testing. We reports PER on the standard core-test set. No language model is used for now.


#### Results
Model                                              |   PER 
|:-| :- |
Transformer (Ours without pre-training)            |   16.8%
