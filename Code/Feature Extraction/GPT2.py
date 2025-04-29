import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import TFGPT2Model, GPT2Tokenizer
import tensorflow as tf


df = pd.read_csv('your/path/to/dataset.csv',encoding='utf-8')
df['content'] = df['content'].astype(str)
train_df, remaining_df = train_test_split(df, test_size=0.4, random_state=42)
val_df, test_df = train_test_split(remaining_df, test_size=0.5, random_state=42)



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




# Specify the path to the local model file
model_path = "your/path/to/GPT2"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = TFGPT2Model.from_pretrained(model_path)



#Data Storage Paths
train_output_file_path = "your/path/to/train_output.npy"
val_output_file_path = "your/path/to/val_output.npy"
test_output_file_path = "your/path/to/test_output.npy"


# Set batch size and maximum sequence length
per = 0
batch_size = 1024
max_length = 128


train_all_outputs = []
for i in range(0, num_train_examples, batch_size):
    train_batch_texts = train_texts[i:i + batch_size]
    train_tokenized_input = tokenizer(train_batch_texts, return_tensors='tf', truncation=True, padding='max_length',
                                      max_length=max_length)
    train_input_ids = train_tokenized_input["input_ids"]

    # train_batch_outputs = bert_model(train_batch_inputs)
    train_outputs = model(train_input_ids)

    train_all_outputs.append(train_outputs[0])

    n = batch_size / num_train_examples
    per += n
    print(f"{per * 100:.2f}%", f"{min(i + batch_size, num_train_examples)} ")

train_output = tf.concat(train_all_outputs, axis=0)
train_output.shape

np.save(train_output_file_path, train_output)

val_all_outputs = []
per = 0
batch_size = 256

for i in range(0, num_val_examples, batch_size):
    val_batch_texts = val_texts[i:i + batch_size]
    val_tokenized_input = tokenizer(val_batch_texts, return_tensors='tf', truncation=True, padding='max_length',
                                    max_length=max_length)
    val_input_ids = val_tokenized_input["input_ids"]

    val_outputs = model(val_input_ids)

    val_all_outputs.append(val_outputs[0])

    n = batch_size / num_val_examples
    per += n
    print(f"{per * 100:.2f}%", f"{min(i + batch_size, num_val_examples)} ")

val_output = tf.concat(val_all_outputs, axis=0)
val_output.shape

np.save(val_output_file_path, val_output)

test_all_outputs = []
per = 0

for i in range(0, num_test_examples, batch_size):
    test_batch_texts = test_texts[i:i + batch_size]
    test_tokenized_input = tokenizer(test_batch_texts, return_tensors='tf', truncation=True, padding='max_length',
                                     max_length=max_length)
    test_input_ids = test_tokenized_input["input_ids"]

    test_outputs = model(test_input_ids)

    test_all_outputs.append(test_outputs[0])

    n = batch_size / num_test_examples
    per += n
    print(f"{per * 100:.2f}%", f"{min(i + batch_size, num_test_examples)} ")

test_output = tf.concat(test_all_outputs, axis=0)
test_output.shape

np.save(test_output_file_path, test_output)