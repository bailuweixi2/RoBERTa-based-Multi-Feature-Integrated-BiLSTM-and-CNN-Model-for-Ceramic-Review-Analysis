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

#Path to the precomputed embeddings required for model training
train_local_output_file_path = "your/path/to/train_output.npy"
val_local_output_file_path = "your/path/to/val_output.npy"
test_local_output_file_path = "your/path/to/test_output.npy"

#Load existing arrays
print("!!")
train_local_output = np.load(train_local_output_file_path)
print("!!")
val_local_output = np.load(val_local_output_file_path)
print("!!")
test_local_output = np.load(test_local_output_file_path)
print("!!")

print(train_local_output.shape)
print(val_local_output.shape)
print(test_local_output.shape)

from tensorflow.keras.layers import Concatenate, Dense, Input
from tensorflow.keras.models import Model

#Same as the max_length set when initially extracting word embeddings.
embedding_dim_1 = 768
max_length = 128

# Assume that the model input dimensions are (sequence length, feature dimension)
input_shape_1 = (max_length, embedding_dim_1)     # local.shape

# Number of labels
num_classes = 2

epochs = 100
es_patience = 3

from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense,Dropout,Conv1D,MaxPool1D,GlobalAveragePooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model

input_1 = Input(shape=input_shape_1)
x2 = Dense(128, activation='relu')(input_1)
dense = Dense(32, activation='relu')(x2)
dense = Dropout(0.3)(dense)
dense = GlobalAveragePooling1D()(dense)
#dense = Flatten()(dense)
output = Dense(num_classes, activation='softmax')(dense)

model = Model(inputs=input_1, outputs=output)
model.summary()

from tensorflow.keras.callbacks import EarlyStopping
batch_size = 128

#If performing multi-class classification, the loss function should be changed to categorical_crossentropy
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_2 = model.fit(train_local_output,train_labels,epochs=epochs,validation_data=(val_local_output,val_labels),
                    callbacks=[EarlyStopping(patience=es_patience)])

loss, accuracy = model.evaluate(test_local_output,test_labels)

print('Test loss:', loss)
print('Test accuracy:', accuracy)

test_pre = model.predict(test_local_output)
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
