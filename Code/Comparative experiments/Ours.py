import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from tensorflow.keras.datasets import imdb


df = pd.read_csv('your/path/to/dataset.csv',encoding='utf-8')
df['content'] = df['content'].astype(str)
train_df, remaining_df = train_test_split(df, test_size=0.4, random_state=42)
val_df, test_df = train_test_split(remaining_df, test_size=0.5, random_state=42)


train_label_counts = train_df['label'].value_counts()
val_label_counts = val_df['label'].value_counts()
test_label_counts = test_df['label'].value_counts()

train_label_counts,val_label_counts,test_label_counts


train_texts = train_df['content'].tolist()
train_labels = train_df['label'].values
num_train_examples = len(train_texts)

val_texts = val_df['content'].tolist()
val_labels = val_df['label'].values
num_val_examples = len(val_texts)

test_texts = test_df['content'].tolist()
test_labels = test_df['label'].values
num_test_examples = len(test_texts)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

print("Label:")
print(train_labels[:10])
print(test_labels[:10])

le = LabelEncoder()
train_labels = le.fit_transform(train_labels).reshape(-1,1)
val_labels = le.fit_transform(val_labels).reshape(-1,1)
test_labels = le.transform(test_labels).reshape(-1,1)
print("LabelEncoder")
print(train_labels[:10])
print(len(train_labels))

ohe = OneHotEncoder()
train_labels = ohe.fit_transform(train_labels).toarray()
val_labels = ohe.fit_transform(val_labels).toarray()
test_labels = ohe.transform(test_labels).toarray()
print("OneHotEncoder:")
print(train_labels[:10])

#Data Storage Paths
train_local_output_file_path = "your/path/to/RoBERT/train_local_output.npy"
train_gobal_output_file_path = "your/path/to/RoBERT/train_gobal_output.npy"
val_local_output_file_path = "your/path/to/RoBERT/val_local_output.npy"
val_gobal_output_file_path = "your/path/to/RoBERT/val_gobal_output.npy"
test_local_output_file_path = "your/path/to/RoBERT/test_local_output.npy"
test_gobal_output_file_path = "your/path/to/RoBERT/test_gobal_output.npy"


print("!!")
train_local_output = np.load(train_local_output_file_path)
print("!!")
train_gobal_output = np.load(train_gobal_output_file_path)
print("!!")
test_local_output = np.load(test_local_output_file_path)
print("!!")
test_gobal_output = np.load(test_gobal_output_file_path)
print("!!")
val_local_output = np.load(val_local_output_file_path)
print("!!")
val_gobal_output = np.load(val_gobal_output_file_path)
print("!!")

train_local_output.shape,val_local_output.shape,test_local_output.shape
train_gobal_output.shape,val_gobal_output.shape,test_gobal_output.shape
#The numbers of training, validation, and test samples vary across different datasets.
train_gobal_output_new = tf.reshape(train_gobal_output, [0, 64, 12])#Replace 0 with the number of samples in your train set.
val_gobal_output_new = tf.reshape(val_gobal_output, [0, 64, 12])#Replace 0 with the number of samples in your test set.
test_gobal_output_new = tf.reshape(test_gobal_output, [0, 64, 12])#Replace 0 with the number of samples in your val set.
train_gobal_output_new.shape,val_gobal_output_new.shape,test_gobal_output_new.shape

print(train_local_output.shape)
print(val_gobal_output_new.shape)
print(val_local_output.shape)
print(val_gobal_output_new.shape)
print(test_local_output.shape)
print(train_gobal_output_new.shape)

from tensorflow.keras.layers import Concatenate, Dense, Input
from tensorflow.keras.models import Model

#Same as the max_length set when initially extracting word embeddings.
embedding_dim_1 = 768
max_length = 128
import tensorflow as tf
from tensorflow.keras.layers import Layer

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        et = tf.keras.backend.squeeze(tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b), axis=-1)
        at = tf.keras.backend.softmax(et, axis=-1)
        at = tf.keras.backend.expand_dims(at, axis=-1)
        output = tf.keras.backend.sum(inputs * at, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

from tensorflow.keras.layers import Concatenate, Dense, Input
from tensorflow.keras.models import Model

embedding_dim_1 = 768
embedding_dim_2 = 64
expand_length = 12

# Assume that the model input dimensions
input_shape_1 = (max_length, embedding_dim_1)     # local.shape
input_shape_2 = (embedding_dim_2, expand_length)  # gobal.shape
num_filters = 64  # Number of convolutional filters

C_filter_size_1 = 5  # Size of convolutional filter 1
C_filter_size_2 = 6  # Size of convolutional filter 2

pool_size_3 = 2
pool_size_4 = 2

B_drop = 0.2
C_drop = 0.1
drop = 0.3

# Number of labels
num_classes = 2

epochs = 100
es_patience = 2

from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense,Dropout,Conv1D,MaxPool1D,GlobalMaxPooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model

input_1 = Input(shape=input_shape_1)
x1 = Bidirectional(LSTM(num_filters, return_sequences=True))(input_1)
x1 = Dropout(B_drop)(x1)
x1 = AttentionLayer()(x1)
#x1 = Dense(128, activation='relu')(x1)

input_2 = Input(shape=input_shape_2)
x2 = Conv1D(32, C_filter_size_1, activation='relu')(input_2)
x2 = MaxPool1D(pool_size=pool_size_3)(x2)
x2 = Conv1D(64, C_filter_size_2, activation='relu')(x2)
x2 = MaxPool1D(pool_size=pool_size_4)(x2)
x2 = Dropout(C_drop)(x2)
x2 = AttentionLayer()(x2)
#x2 = Dense(128, activation='relu')(x2)

concatenated = Concatenate()([x1,x2])
dense = Dense(32, activation='relu')(concatenated)
dense = Dropout(drop)(dense)
output = Dense(num_classes, activation='sigmoid')(dense)

model = Model(inputs=[input_1,input_2], outputs=output)
model.summary()

from tensorflow.keras.callbacks import EarlyStopping
batch_size = 128

#If performing multi-class classification, the loss function should be changed to categorical_crossentropy
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_2 = model.fit([train_local_output,train_gobal_output_new],train_labels,
                    epochs=epochs,validation_data=([val_local_output,val_gobal_output_new],val_labels),callbacks=[EarlyStopping(patience=es_patience)])

loss, accuracy = model.evaluate([test_local_output,test_gobal_output_new],test_labels)

print('Test loss:', loss)
print('Test accuracy:', accuracy)

test_pre = model.predict([test_local_output,test_gobal_output_new])
predicted_labels = np.argmax(test_pre, axis=1)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

#print("Label:")
#print(predicted_labels[:10])

le = LabelEncoder()
predicted_labels = le.fit_transform(predicted_labels).reshape(-1,1)

#print("LabelEncoder")
#print(predicted_labels[:10])
#print(len(predicted_labels))

ohe = OneHotEncoder()
predicted_labels = ohe.fit_transform(predicted_labels).toarray()
#print("OneHotEncoder:")
#print(predicted_labels[:10])

from sklearn.metrics import classification_report
#If performing multi-class classification, new classes need to be added.
target_names = ['class 0', 'class 1']
report = classification_report(test_labels, predicted_labels, target_names=target_names, digits=5)
print(report)

