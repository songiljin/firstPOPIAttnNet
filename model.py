import tensorflow as tf
from tensorflow.keras.layers import GRU


class BahdanauAttention(tf.keras.layers.Layer):
    
  def __init__(self, w1_units):
      
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(w1_units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, values):
      
    # 分数的形状 == （批大小，pep_length，1）
    # 我们在最后一个轴上得到 1， 因为我们把分数应用于 self.V
    # 在应用 self.V 之前，张量的形状是（批大小，pep_length，单位）
    score = self.V(tf.nn.tanh(
        self.W1(values)))

    # 注意力权重 （attention_weights） 的形状 == （批大小，pep_length，1）
    attention_weights = tf.nn.softmax(score, axis=1)

    # 上下文向量 （context_vector） 求和之后的形状 == （批大小，隐藏层大小）
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class POPIAttnNet(tf.keras.Model):
    
  def __init__(self,w1_units,MHCIP_units,batch_sz,fc_dim):
      
    super(POPIAttnNet, self).__init__()
    self.batch_sz=batch_sz
    self.w1_units=w1_units
    self.fc_dim=fc_dim
    self.MHCIP_units = MHCIP_units
    self.gru1_forward = tf.keras.layers.GRU(self.MHCIP_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
                                              
    self.gru1_backward = tf.keras.layers.GRU(self.MHCIP_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform',
                                   go_backwards=True)
    self.bi_gru1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
                 self.MHCIP_units,return_sequences=True,return_state=True,dropout=0.5,
                 recurrent_dropout=0.5,recurrent_initializer='glorot_uniform',kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        input_shape=(None,9,553))
               
                               
    self.bi_gru2 = tf.keras.layers.Bidirectional( tf.keras.layers.GRU(
                 256,return_sequences=True,dropout=0.5,
                 recurrent_dropout=0.5)
               
                               )   
    self.bi_gru3 = tf.keras.layers.Bidirectional( tf.keras.layers.GRU(
                 256,return_sequences=True,dropout=0.5,
                 recurrent_dropout=0.5)
               
                               )   
    self.dropout1 = tf.keras.layers.Dropout(0.2)
    self.dropout2 = tf.keras.layers.Dropout(0.5)
    self.dropout3 = tf.keras.layers.Dropout(0.5)
                                              
    self.fc1 = tf.keras.layers.Dense(fc_dim,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001))
    self.fc2 = tf.keras.layers.Dense(2)

    # 用于注意力
    self.attention = BahdanauAttention(self.w1_units)

  def call(self, peptide,hidden):
      
    # 编码器输出 （enc_output） 的形状 == （批大小，最大长度，隐藏层大小）
    output, final_memory_state_1,_ = self.bi_gru1(peptide,initial_state = [hidden,hidden])
    #output, final_memory_state_1,_ = self.bi_gru1(peptide)
    output = self.dropout1(output)
    output = self.bi_gru2(output)
    #output = self.dropout2(output)
    output = self.bi_gru3(output)
    
    context_vector, attention_weights = self.attention(output)

    output = self.fc1(context_vector)
    output = self.dropout3(output)
    fc2_output = self.fc2(output)
    # 输出的形状 == （批大小，vocab）

    return fc2_output,attention_weights
    
  def initialize_hidden_state(self):
      
    return tf.zeros((self.batch_sz, self.MHCIP_units))