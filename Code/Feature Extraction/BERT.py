import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import TFBertModel
from transformers import BertTokenizer
import tensorflow as tf


df = pd.read_csv('your/path/to/dataset.csv',encoding='utf-8')
df['content'] = df['content'].astype(str)
train_df, remaining_df = train_test_split(df, test_size=0.4, random_state=42)
val_df, test_df = train_test_split(remaining_df, test_size=0.5, random_state=42)

# Specify the path to the local model file
model_name = 'your/path/to/bert-base-uncased/'
bert_model = TFBertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)


train_label_counts = train_df['label'].value_counts()
val_label_counts = val_df['label'].value_counts()
test_label_counts = test_df['label'].value_counts()
train_label_counts,val_label_counts,test_label_counts


# Set batch size and maximum sequence length
batch_size = 1024
max_length = 64

# Define helper functions for encoding and vector transformation
def encode_and_convert_to_tensors(texts):
    encoded = tokenizer.batch_encode_plus(
        texts,
        pad_to_max_length=True,
        max_length=max_length,
        return_tensors='tf'
    )
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

#Data Processing
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
train_local_output_file_path = "your/path/to/train_local_output.npy"
train_gobal_output_file_path = "your/path/to/train_gobal_output.npy"
val_local_output_file_path = "your/path/to/val_local_output.npy"
val_gobal_output_file_path = "your/path/to/val_gobal_output.npy"
test_local_output_file_path = "your/path/to/test_local_output.npy"
test_gobal_output_file_path = "your/path/to/test_gobal_output.npy"


train_all_outputs_0 = []
train_all_outputs_1 = []
per = 0
for i in range(0, num_train_examples, batch_size):
    train_batch_texts = train_texts[i:i + batch_size]
    train_batch_inputs = encode_and_convert_to_tensors(train_batch_texts)

    train_batch_outputs = bert_model(train_batch_inputs)

    train_all_outputs_0.append(train_batch_outputs[0])
    train_all_outputs_1.append(train_batch_outputs[1])

    n = batch_size / num_train_examples
    per += n
    print(f"{per * 100:.2f}%")

train_local_output = tf.concat(train_all_outputs_0, axis=0)
train_gobal_output = tf.concat(train_all_outputs_1, axis=0)
train_local_output.shape,train_gobal_output.shape

np.save(train_local_output_file_path, train_local_output)
np.save(train_gobal_output_file_path, train_gobal_output)


val_all_outputs_0 = []
val_all_outputs_1 = []
per = 0
for i in range(0, num_val_examples, batch_size):
    val_batch_texts = val_texts[i:i + batch_size]
    val_batch_inputs = encode_and_convert_to_tensors(val_batch_texts)

    val_batch_outputs = bert_model(val_batch_inputs)

    val_all_outputs_0.append(val_batch_outputs[0])
    val_all_outputs_1.append(val_batch_outputs[1])

    n = batch_size / num_val_examples
    per += n
    print(f"{per * 100:.2f}%")

val_local_output = tf.concat(val_all_outputs_0, axis=0)
val_gobal_output = tf.concat(val_all_outputs_1, axis=0)
val_local_output.shape,val_gobal_output.shape


np.save(val_local_output_file_path, val_local_output)
np.save(val_gobal_output_file_path, val_gobal_output)


test_all_outputs_0 = []
test_all_outputs_1 = []
per = 0
for i in range(0, num_test_examples, batch_size):
    test_batch_texts = test_texts[i:i + batch_size]
    test_batch_inputs = encode_and_convert_to_tensors(test_batch_texts)

    test_batch_outputs = bert_model(test_batch_inputs)

    test_all_outputs_0.append(test_batch_outputs[0])
    test_all_outputs_1.append(test_batch_outputs[1])

    n = batch_size / num_test_examples
    per += n
    print(f"{per * 100:.2f}%")

test_local_output = tf.concat(test_all_outputs_0, axis=0)
test_gobal_output = tf.concat(test_all_outputs_1, axis=0)
test_local_output.shape,test_gobal_output.shape

np.save(test_local_output_file_path, test_local_output)
np.save(test_gobal_output_file_path, test_gobal_output)