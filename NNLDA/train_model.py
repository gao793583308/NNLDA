import pandas as pd
import numpy as np
import random
import tensorflow as tf

def produce_negative_data(data,sample_num,num_lncRNA):
    negative_data = pd.DataFrame(columns=['ncRNA Symbol','Disease Name','lable'])
    total_lncRNA = set(i for i in range(num_lncRNA))
    grouped = data.groupby('Disease Name')
    for name,group in grouped:
        lncRNA = set(group['ncRNA Symbol'])
        num_positive = len(lncRNA)
        disease = set(group['Disease Name'])
        lncRNA = total_lncRNA - lncRNA
        df = pd.DataFrame(columns=['ncRNA Symbol','Disease Name','lable'])
        df['ncRNA Symbol'] = random.sample(lncRNA,min(num_positive*sample_num,len(lncRNA)))
        df['Disease Name'] = list(disease)*min(num_positive*sample_num,len(lncRNA))
        df['lable'] = [0] * min(num_positive*sample_num,len(lncRNA))
        negative_data = negative_data.append(df,ignore_index=True)
    return data.append(negative_data,ignore_index=True)

data = pd.read_csv("./data/all_data.csv")
use_data = data[['ncRNA Symbol','ncRNA Category','Species','Disease Name']]
use_data = use_data[use_data['ncRNA Category'] == 'lncRNA']
use_data = use_data[use_data['Species'] == 'Homo sapiens']
use_data = use_data[['ncRNA Symbol','Disease Name']]
use_data.drop_duplicates(inplace=True)

lncRNA_shift = {}
disease_shift = {}
cnt_lncRNA = 0
for each_lncRNA in sorted(list(set(use_data['ncRNA Symbol']))):
    lncRNA_shift[each_lncRNA] = cnt_lncRNA
    cnt_lncRNA += 1

cnt_disease = 0
for each_disease in sorted(list(set(use_data['Disease Name']))):
    disease_shift[each_disease] = cnt_disease
    cnt_disease += 1

use_data['ncRNA Symbol'] = use_data['ncRNA Symbol'].apply(lambda x:lncRNA_shift[x])
use_data['Disease Name'] = use_data['Disease Name'].apply(lambda x:disease_shift[x])
use_data['lable'] = 1
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
lable = tf.placeholder('float32',[None,1])


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
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction,labels=lable)
loss = tf.reduce_mean(loss)


regular_decay = 0.01
loss = loss + regular_decay*tf.reduce_mean(tf.reduce_sum(tf.square(lncRNA_embeddings),axis=1))
loss = loss + regular_decay*tf.reduce_mean(tf.reduce_sum(tf.square(disease_embeddings),axis=1))
loss = loss + tf.contrib.layers.l2_regularizer(regular_decay)(weights['layer_prediction'])
for i in range(len(deep_layers)):
    loss = loss + tf.contrib.layers.l2_regularizer(regular_decay)(weights["layer_%d" % i])

learning_rate = 0.01
optimizer = tf.train.AdamOptimizer(learning_rate)
global_step = tf.Variable(0, trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)
saver = tf.train.Saver()

#-----------------------------------------------------------------------------------
epoch = 100
batch_size = 1024
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tran_data = use_data
    #------------------------------------------------------------
    negative_record = {}
    for data,row in tran_data.iterrows():
        if(row['Disease Name'] not in negative_record):
            negative_record[row['Disease Name']] = set()
        negative_record[row['Disease Name']].add(row['ncRNA Symbol'])
    total_lncRNA = set(i for i in range(cnt_lncRNA))
    for i in negative_record.keys():
        negative_record[i] = total_lncRNA - negative_record[i]
    #-----------------------------------------------------------------
    tran_data = produce_negative_data(tran_data, 4, cnt_lncRNA)
    for each_epoch in range(epoch):
        temp = []
        tran_data = tran_data.sample(frac=1)
        begin = 0
        while(begin < len(tran_data)):
            batch_data = tran_data.iloc[begin:begin + batch_size]
            #print(batch_data['ncRNA Symbol'])
            _,t1= sess.run([train_op,loss],
              feed_dict={
                  lncRNA_index:batch_data['ncRNA Symbol'],
                  disease_index:batch_data['Disease Name'],
                  lable:np.reshape(np.array(batch_data['lable']),newshape=[-1,1])
              })
            begin = begin + batch_size
            temp.append(t1)
        print("epoch:%d train_loss:%lf" %(each_epoch,sum(temp)/len(temp)))
    print("save model to /model/model.ckpt")
    saver.save(sess,"./model/model.ckpt")