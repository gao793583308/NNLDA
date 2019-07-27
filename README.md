Title: Deep learning enables accurate prediction of interplay between lncRNA and disaeas 
====================
###########################################################################################

1.data

The experimental data are downloaded from [LncRNADiseas2.0 database](http://www.rnanut.net/lncrnadisease/)

2. Implementation

NNLDA is implemented in Python 3.5 and use Tensorflow 1.12.0. Package numpy and pandas are also required.

###########################################################################################
## This section introduces how to run NNLDA to make prediction
### Train model(model will save in ./model)
      python train_model.py
### make prediciton
      python make_prediction.py
Then you need to enter the name of the disease you need to predict. A scoring file will be generater when the prediciton is completed
