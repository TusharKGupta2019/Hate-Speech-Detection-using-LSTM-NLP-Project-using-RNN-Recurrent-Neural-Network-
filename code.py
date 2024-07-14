# Importing Libararies
import pandas as pd
import spacy
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('labeled_data.csv')

df.shape

df.head()

df.columns

# Deleting unwanted columns
df.drop(columns = ['Unnamed: 0','count','hate_speech','offensive_language','neither'], inplace = True)

df.head()

# checking null values
df.isnull().sum()

df['tweet'].iloc[0]

df['tweet'].iloc[100]

df['tweet'].iloc[1000]

# Deleting unwanted symbols and numeric data
df['processed_tweet']=df['tweet'].str.replace(r'[^a-zA-Z]',' ', regex=True)

df.head()

# Handling unwanted spaces
df['processed_tweet_2']=df['processed_tweet'].str.replace(r'[\s]+',' ',regex = True)

df.head()

df['processed_tweet_2'].iloc[1000]

df.drop(columns = ['tweet','processed_tweet'], inplace = True)

df.head()

# NLP
nlp = spacy.load('en_core_web_sm')

# lemmatization
def lemmatization(text):
  doc = nlp(text)
  lemmalist = [word.lemma_ for word in doc]
  return ' '.join(lemmalist)

df['lemma_tweet']=df['processed_tweet_2'].apply(lemmatization)

df.head()

# Removing the stopwords
def remove_stopwords(text):
  doc = nlp(text)
  no_stopwords_list = [word.text for word in doc if not word.is_stop]
  return ' '.join(no_stopwords_list)

df['final_tweet']=df['lemma_tweet'].apply(remove_stopwords)

df.head()

df.drop(columns = ['processed_tweet_2', 'lemma_tweet'], inplace = True)

df.head()

# One - hot representation
vocab_size = 10000
one_hot_representation = [one_hot(words, vocab_size) for words in df['final_tweet']]

df['final_tweet'].iloc[0]

one_hot_representation[0]

for i in range(0, 4):
  print(df['final_tweet'].iloc[i])

for i in range(0, 4):
  print(one_hot_representation[i])

sentence_length = 20
embedded_tweet = pad_sequences(one_hot_representation, padding = 'pre', maxlen = sentence_length)

for i in range(0, 4):
  print(embedded_tweet[i])

X = np.array(embedded_tweet)
y = np.array(df['class'])

df['class'].value_counts()

smote = SMOTE(sampling_strategy='minority')
X,y = smote.fit_resample(X,y)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X.shape, X_train.shape, X_test.shape

# Creating Model

dimension = 50

model = keras.Sequential([
    # embedding layer
    keras.layers.Embedding(vocab_size, dimension, input_length = sentence_length),
    # LSTM layers (stacked)
    keras.layers.LSTM(100, return_sequences = True),
    keras.layers.LSTM(50, return_sequences = True),
    keras.layers.LSTM(50),
    # Output Layer
    keras.layers.Dense(3, activation = 'softmax')
]) 

model.compile(optimizer = 'adam',
               loss = 'sparse_categorical_crossentropy',
               metrics = ['accuracy'])
    

model.summary()

model.fit(X_train, y_train, epochs = 10, batch_size = 32)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Model Accuracy : {accuracy * 100}')

pred = np.argmax(model.predict(X_test), axis = -1)

y_test[:5]

pred[:5]

print(classification_report(y_test, pred))

cf = confusion_matrix(y_test, pred, normalize = 'true')
sns.heatmap(cf, annot = True, cmap = 'crest')
plt.xlabel('PREDICTED')
plt.ylabel('ACTUAL')
plt.show()

