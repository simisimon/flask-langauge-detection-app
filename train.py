import sys
import os
import requests

import nltk
import sklearn
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np


nltk.download('stopwords')
nltk.download('punkt')

print(tf.__version__)

# Download data and save it into the data folder
DATA_DIRECTORY = "data"

with open(f"{DATA_DIRECTORY}/train.csv", "wb") as f:
    f.write(requests.get("https://huggingface.co/datasets/papluca/language-identification/resolve/main/train.csv").content)

with open(f"{DATA_DIRECTORY}/test.csv", "wb") as f:
    f.write(requests.get("https://huggingface.co/datasets/papluca/language-identification/resolve/main/test.csv").content)

with open(f"{DATA_DIRECTORY}/valid.csv", "wb") as f:
    f.write(requests.get("https://huggingface.co/datasets/papluca/language-identification/resolve/main/valid.csv").content)

# Load data in pandas for filtering

train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/valid.csv")
test_df = pd.read_csv("data/test.csv") 

# Select only "en", "es" and "de"
lang_list = ["es", "en", "de"]

train_df = train_df.loc[train_df.labels.isin(lang_list)]
val_df = val_df.loc[val_df.labels.isin(lang_list)]
test_df = test_df.loc[test_df.labels.isin(lang_list)]

# We calculate a list of words (tokens) which will be used for evaluate the distribution of
# tokens in the dataset

# We declare a stoplist for the three used languages
stoplist = [nltk.corpus.stopwords.words(lang) for lang in ["english", "spanish", "german"]]
stoplist = set([word for lang_list in stoplist for word in lang_list])

# Now, only for visualization purposes we create a listh with all the tokens
word_list = []
for sentence in train_df["text"].to_list():
    word_list += [word for word in nltk.tokenize.word_tokenize(sentence) if word not in stoplist]

print(f"Our corpus consists of {len(word_list)} different words.")
print("Distribution of labels:")
train_df.labels.value_counts()


splitted_text = train_df['text'].apply(lambda txt: txt.split(' '))
splitted_text_len = splitted_text.apply(lambda x: len(x))
ax = splitted_text_len.plot.hist(bins=12, alpha=0.5)

print("Average length: {}".format(splitted_text_len.mean()))
print("Maximum length: {}".format(splitted_text_len.max()))
print("Standard deviation length: {}".format(splitted_text_len.std()))

le = sklearn.preprocessing.LabelEncoder()
le.fit(lang_list)

num_classes = len(le.classes_)

train_labels = tf.keras.utils.to_categorical(le.transform(train_df.pop('labels')), num_classes=num_classes)
val_labels = tf.keras.utils.to_categorical(le.transform(val_df.pop('labels')), num_classes=num_classes)
test_labels = tf.keras.utils.to_categorical(le.transform(test_df.pop('labels')), num_classes=num_classes)

raw_train_ds = tf.data.Dataset.from_tensor_slices((train_df["text"].to_list(), train_labels)) # X, y
raw_val_ds = tf.data.Dataset.from_tensor_slices((val_df["text"].to_list(), val_labels))
raw_test_ds = tf.data.Dataset.from_tensor_slices((test_df["text"].to_list(), test_labels))

batch_size = 32
seed = 42

for text, label in raw_train_ds.take(1):
    print("Text: ", text.numpy())
    print("Label:", label.numpy())

# Prepare dataset for training

max_features = 10000 # top 10K most frequent words
sequence_length = 50 # Was defined in the previous data exploration section

vectorize_layer = tf.keras.layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# This step is important to perform it only on the training set. The `adapt()` method will learn the vocabulary for our dataset

vectorize_layer.adapt(train_df["text"].to_list()) # vectorize layer is fitted to the training data

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# Reviewing how a sample of the corpus will be fed to the model

text_batch, label_batch = next(iter(raw_train_ds.batch(64)))
first_review, first_label = text_batch[0], label_batch[0]
print("First text: ", first_review)
print("Language (label)", le.inverse_transform([np.argmax(first_label)]))
print("Vectorized text", vectorize_text(first_review, first_label))

train_ds = raw_train_ds.map(lambda x, y: (vectorize_layer(x), y)) # returns vectorize_layer(text), label
val_ds = raw_val_ds.map(lambda x, y: (vectorize_layer(x), y))
test_ds = raw_test_ds.map(lambda x, y: (vectorize_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.batch(batch_size=batch_size)
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.batch(batch_size=batch_size)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.batch(batch_size=batch_size)
test_ds = test_ds.prefetch(AUTOTUNE)

 # Model training

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_features + 1, 16),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3)])

model.summary()

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10)

loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

# Plot loss 
history_dict = history.history
print("Keys of the history variable")
print(history_dict.keys())

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

def softmax(x, axis=None):
    """
    :param x: numpy array where each row corresponds to the scores (outputs) of the mode (e.g.: [[0.2, 0.6, 0.7], [0.1, 0.5, 0.4]]) 
    :param axis: axis where to compute values along (e.g., axis=1 computes the softmax along the second axis, i.e., the rows)
    :return: an array of the same shape as `x`. The result will sum to 1 along the specified axis.
    """
    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

 
examples = [
  "Esto me pareció increíble",
  "I think it was incredible.",
  "Ich finde es war unglaublich"
]

examples_vectorized = vectorize_layer(examples)

logits = model.predict(examples_vectorized)
probits = softmax(logits, axis=1)
idx_predictions = np.argmax(logits, axis=1)
print("Probabilities: {}".format(np.max(probits, axis=1)))
print("Corresponding classes: {}".format(le.inverse_transform(idx_predictions)))

model.save('models/saved_model/simple_mlp_novectorize.h5')

 
def store_text_vectorizer(vectorizer, file_path: str):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    model.add(vectorizer)
    model.save(file_path)

store_text_vectorizer(vectorize_layer, "../data/vectorizer")

 # Load the model
saved_model = tf.keras.models.load_model('models/saved_model/simple_mlp_novectorize.h5')
vectorize_layer = tf.keras.models.load_model('models/vectorizer').layers[0]

new_examples = ["I made it to task 3.", "Ich hab es bis task 3 geschafft."]

new_examples_vectorized = vectorize_layer(new_examples)
logits = saved_model.predict(new_examples_vectorized)

probabilities = softmax(logits, axis=1)
print(probabilities)
for label, text in zip(probabilities, new_examples):
    print(text, le.inverse_transform([np.argmax(label)]))