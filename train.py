import numpy as np 
import os
import time
import config
import tensorflow as tf
from make_dataset import Dataset
from model import POPIAttnNet,BahdanauAttention
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, confusion_matrix
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import  OneHotEncoder



class_dataset=Dataset()
train_dataset,val_dataset,test_dataset=class_dataset.train_val_test_dataset()
#train_peptides,train_labels,positive_length,val_peptides,val_labels,test_peptides,test_labels=    mhci_data()
model=  POPIAttnNet(config.w1_units,config.mhcip_units,config.BATCH_SIZE,config.fc_dim)
hidden=model.initialize_hidden_state()

optimizer = tf.keras.optimizers.Adam(lr=config.lr)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  #mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  #mask = tf.cast(mask, dtype=loss_.dtype)
  #loss_ *= mask

  return tf.reduce_mean(loss_)

def accuary(prediction,label):
    i=0
    j=0
    prediction=tf.argmax(prediction,axis=-1)
    for p in prediction:
        if p.numpy()==label[i].numpy():
            j+=1
        i+=1
    
    return j

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 model=model)

EPOCHS = config.EPOCHS 

@tf.function
def train_step(peptides,labels,hidden):
  
  loss = 0
  with tf.GradientTape() as tape:
    prediction,attention_weights = model(peptides, hidden)
    loss += loss_function(labels, prediction)

    variables = model.trainable_variables 

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return loss,prediction

EPOCHS = config.EPOCHS
steps_per_epoch = len(train_dataset)

for epoch in range(EPOCHS):
  start = time.time()

  hidden = model.initialize_hidden_state()
  total_loss = 0
  acc_score =0

  for (batch, (peptides, labels)) in enumerate(train_dataset.take(steps_per_epoch)):
    batch_loss,prediction = train_step(peptides, labels, hidden)
    total_loss += batch_loss
    
    acc = accuracy_score(labels,tf.argmax(prediction,axis=-1))
    acc_score+=acc
    if batch % 10 == 0:
        print('Epoch {} Batch {} Loss {:.4f} Acc {:.2f}'.format(epoch + 1,
                                                     batch,
                                                     batch_loss,acc))
  # 每 2 个周期（epoch），保存（检查点）一次模型
  if (epoch + 1) % 2 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)

  print('Epoch {} Loss {:.4f} Acc {:.4f}'.format(epoch + 1,
                                      total_loss/steps_per_epoch,acc_score/steps_per_epoch))
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
  
  total_num = 0
  total_correct = 0
  count = 0
  total_acc = 0
  batch_sz=64
  MHCIP_units=config.mhcip_units 
  enc=OneHotEncoder()
  y_actual_val = list()
  y_pred_val = list()
  for x,y in val_dataset:

            count+=1
            hidden=tf.zeros((batch_sz, MHCIP_units))
            logits,attention_weight = model(x,hidden)
            #accuary_score = accuary(logits,y) 
            
            y_expand = tf.expand_dims(y,axis=1)
            #enc.fit(y_expand)
            y_expand = enc.fit_transform(y_expand).toarray()
            y_actual_val += list(y_expand)
            
            prob = tf.nn.softmax(logits,axis=1)
            y_pred_val += list(prob)
            #roc_auc = roc_auc_score(y_expand,prob)
            #total_roc_auc = total_roc_auc+roc_auc
            
            pred = tf.argmax(prob,axis=1)
            pred = tf.cast(pred,dtype=tf.int32)
            
            accuracy = accuracy_score(y,pred)
            total_acc = total_acc+accuracy

            correct = tf.cast(tf.equal(pred,y),dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += correct
            acc = total_correct/total_num
  print(attention_weight[-1])
  roc_auc = roc_auc_score(y_actual_val,y_pred_val)
  prc_auc = average_precision_score(y_actual_val,y_pred_val)
  print("acc:",acc,"acc:","accuary:", total_acc/count,"roc_auc:",roc_auc,"prc_auc:",prc_auc)
  #print("precision:",precision,"recall:",recall,"f1:",f1,"roc_auc:",roc_auc,"prc_auc:",prc_auc)
  #print("tn,fp,fn,tp",tn,fp,fn,tp)
  
  total_num = 0
  total_correct = 0
  batch_sz=64
  MHCIP_units=config.mhcip_units 
  y_actual_test = list()
  y_pred_test = list()
  pred_t = list()
  label_t = list()
  for x,y in test_dataset:
            label_t += list(y)
            hidden=tf.zeros((batch_sz, MHCIP_units))
            logits,attention_weight = model(x,hidden)
            
            y_expand = tf.expand_dims(y,axis=1)
            y_expand = enc.fit_transform(y_expand).toarray()
            y_actual_test += list(y_expand)
            
            accuary_score = accuary(logits,y) 
            prob = tf.nn.softmax(logits,axis=1)
            y_pred_test += list(prob)
            pred = tf.argmax(prob,axis=1)
            pred = tf.cast(pred,dtype=tf.int32)
            pred_t += list(pred)

            correct = tf.cast(tf.equal(pred,y),dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += correct
            acc = total_correct/total_num
  print("test_acc:",acc,"acc:",accuary_score/64 )    
  acc_scores = accuracy_score(label_t,pred_t)
  precision_scores = precision_score(label_t,pred_t)
  recall_scores = recall_score(label_t,pred_t)
  F1_scores = f1_score(label_t,pred_t)
  roc_auc = roc_auc_score(y_actual_test,y_pred_test)
  prc_auc = average_precision_score(y_actual_test,y_pred_test)
  print("test_auc:",roc_auc,"prc:",prc_auc,"acc_score:",acc_scores)
  print("pre_score:",precision_scores,"recall_score:",recall_scores,"F1score:",F1_scores)
  
