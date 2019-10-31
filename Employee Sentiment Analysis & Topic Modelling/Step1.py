from initial import df

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re
import numpy as np
from collections import Counter

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
%matplotlib inline

sentiment_columns = ['postitives', 'negatives']
df = df[sentiment_columns].copy()


def clean(doc):
  '''
  Function to clean the document (remove stop words, punctuations, lemmatize each word)
  '''
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    processed = re.sub(r"\d+","",normalized)
    y = processed.split()
    return y

# Store the cleaned sentences (POSITIVE & NEGATIVE) in different lists
train_clean_positive_sentences, train_clean_negative_sentences = [], []
for pos_line, neg_line in zip(df['positives'], df['negatives']):
    line1, line2 = pos_line.strip(), neg_line.strip()
    cleaned1, cleaned2 = clean(line1), clean(line2)
    cleaned1, cleaned2 = ' '.join(cleaned1), ' '.join(cleaned2)
    train_clean_positive_sentences.append(cleaned1)
    train_clean_negative_sentences.append(cleaned2)

vectorizer = TfidfVectorizer(stop_words='english')
pos_X = vectorizer.fit_transform(train_clean_positive_sentences)
neg_X = vectorizer.fit_transform(train_clean_negative_sentences)


for val, title in zip(['positives', 'negatives'], ['Postive Reviews', 'Negative Reviews']): 
  text = list(df[val])
  text = ''.join(text)
  #  Visualize the Word cloud
  stop = set(stopwords.words('english'))
  wordcloud = WordCloud(width=1000,height=750,stopwords=stop,max_font_size=70, max_words=100, background_color="white").generate(text)
  plt.figure(figsize=(20,20))
  plt.imshow(wordcloud, interpolation="bilinear")
  plt.title(title, fontsize=24)
  plt.axis("off")
  plt.savefig(title+'.jpg')
print('Figures Saved')
