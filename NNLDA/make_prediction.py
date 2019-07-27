import pandas as pd
import numpy as np
import random
import tensorflow as tf

data = pd.read_csv("./data/all_data.csv")
use_data = data[['ncRNA Symbol','ncRNA Category','Species','Disease Name']]
use_data = use_data[use_data['ncRNA Category'] == 'lncRNA']
use_data = use_data[use_data['Species'] == 'Homo sapiens']
use_data = use_data[['ncRNA Symbol','Disease Name']]
use_data.drop_duplicates(inplace=True)

lncRNA_shift = {}
lncRNA_name = []
disease_shift = {}
cnt_lncRNA = 0
for each_lncRNA in sorted(list(set(use_data['ncRNA Symbol']))):
    lncRNA_shift[each_lncRNA] = cnt_lncRNA
    lncRNA_name.append(each_lncRNA)
    cnt_lncRNA += 1

cnt_disease = 0
for each_disease in sorted(list(set(use_data['Disease Name']))):
    disease_shift[each_disease] = cnt_disease
    cnt_disease += 1

#-------------------------- ini NN --------------------------------------------------
EMBEDDING_SIZE = 32
weights = {}
weights['lncRNA'] = tf.get_variable(
    name='lncRNA_embedding',
    dtype=tf.float32,
    initializer=tf.glorot_normal_initializer(),
    shape=[cnt_lncRNA, EMBEDDING_SIZE]
)
weights['disease'] = tf.get_variable(
    name='disease_embedding',
    dtype=tf.float32,
    initializer=tf.glorot_normal_initializer(),
    shape=[cnt_disease, EMBEDDING_SIZE]
)

lncRNA_index = tf.placeholder('int32',[None])
disease_index = tf.placeholder('int32',[None])

lncRNA_embeddings = tf.nn.embedding_lookup(
    weights['lncRNA'],
    lncRNA_index
)
disease_embeddings = tf.nn.embedding_lookup(
    weights['disease'],
    disease_index
)
#------------------concat part------------------------------------------------------
deep_layers= [32,16,8]
weights['layer_0'] = tf.get_variable(
    name='layer_0',
    dtype=tf.float32,
    initializer=tf.glorot_normal_initializer(),
    shape=[2*EMBEDDING_SIZE,deep_layers[0]]
)
weights['bias_0'] = tf.get_variable(
    name='bias_0',
    dtype=tf.float32,
    initializer=tf.constant_initializer(0.0),
    shape=[1,deep_layers[0]]
)
for i in range(1,len(deep_layers)):
    weights["layer_%d" % i]  = tf.get_variable(
        name='layer_%d' % i,
        dtype=tf.float32,
        initializer=tf.glorot_normal_initializer(),
        shape=[deep_layers[i-1],deep_layers[i]]
    )
    weights["bias_%d" % i] = tf.get_variable(
        name='bias_%d' % i,
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        shape=[1, deep_layers[i]]
    )
y_deep = tf.concat([lncRNA_embeddings,disease_embeddings],axis=1)
for i in range(len(deep_layers)):
    y_deep = tf.add(tf.matmul(y_deep,weights["layer_%d" % i]), weights["bias_%d" % i])
    y_deep = tf.nn.relu(y_deep)
#-----------------inner production--------------------------------------------------
product = tf.multiply(lncRNA_embeddings,disease_embeddings)
predict_vector = tf.concat([product,y_deep],axis=1)
weights['layer_prediction'] = tf.get_variable(
    name='layer_prediction',
    dtype=tf.float32,
    initializer=tf.glorot_normal_initializer(),
    shape=[deep_layers[-1]+EMBEDDING_SIZE,1]
)
weights['bias_prediction'] = tf.get_variable(
    name='bias_prediction',
    dtype=tf.float32,
    initializer=tf.constant_initializer(0.0),
    shape=[1,1]
)
prediction = tf.add(tf.matmul(predict_vector,weights['layer_prediction']), weights['bias_prediction'])
final = tf.sigmoid(prediction)
saver = tf.train.Saver()


#-----------------------------------------------------------------------------------
with tf.Session() as sess:
    saver.restore(sess,"./model/model.ckpt")
    print("Please enter the name of the disease you need to predict")
    diseas_name = input()
    if(diseas_name not in disease_shift):
        print("error: Incorrect disease name")
    else:
        print("Begin predict")
        cnt = disease_shift[diseas_name]
        test_result = sess.run(final,
                   feed_dict={
                       lncRNA_index: [i for i in range(cnt_lncRNA)],
                       disease_index: [cnt]*cnt_lncRNA,
                   })
        score = [[i,test_result[i][0]] for i in range(len(test_result))]
        score = sorted(score,key=lambda x:x[1],reverse=True)
        result_file = diseas_name+".csv"
        file = open(result_file,"w")
        for ind,sc in score:
            file.write(str(lncRNA_name[ind])+","+str(sc)+"\n")
        print("Predict success")

