License
====================
PROGRAM: NNLDA

AUTHOR: Jialu Hu and Yiqun Gao

EMAIL: jhu@nwpu.edu.cn, yiqun.gao@nwpu-bioinformatics.com

Copyright (MTLAB) <2018> 

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.

Title: Deep learning enables accurate prediction of interplay between lncRNA and disaeas 
====================
###########################################################################################

1.data

The experimental data are downloaded from [LncRNADiseas2.0 database](http://www.rnanut.net/lncrnadisease/)

###########################################################################################
This section introduces how to run NNLDA to make prediction
### Train model(model will save in ./model)
      python  	train_model.py
### make prediciton
      python  	make_prediction.py
      Then you need to enter the name of the disease you need to predict.
      A scoring file will be generater when the prediciton is completed
