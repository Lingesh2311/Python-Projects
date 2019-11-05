# Gensim & nltk imports
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import re
import warnings
import sys
import pandas as pd

path = 'data/train.csv'
df = pd.read_csv(path)

# Wordcloud imports
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
# ! %matplotlib inline

# Final LDA model
from pprint import pprint
from time import time
from gensim.test.utils import datapath
import gensim.models.ldamodel as lda

# !{sys.executable} -m spacy download en
warnings.filterwarnings("ignore")
stop = stopwords.words('english')
stop.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_',\
                   'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice',\
                   'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want',\
                   'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])


def sent_to_words(sentences):
    for sent in sentences:
        sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
        sent = re.sub('\s+', ' ', sent)  # remove newline chars
        sent = re.sub("\'", "", sent)  # remove single quotes
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
        yield(sent)  


def create_data_words(val):
  #  Each review is taken and split into words after cleaning and preprocessing
  data = df[val].tolist()
  data_words = list(sent_to_words(data))
  print(f'Sample sentence from {val} reviews')
  print(data_words[:1]) # A sample from the dataset - Positive Reviews
  return data_words


# Building the Latent Dirichilet Allocation Model
# Bigram and Trigram models for the Reviews
def generate_gram_model(data_words):
  bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
  trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
  bigram_mod = gensim.models.phrases.Phraser(bigram)
  trigram_mod = gensim.models.phrases.Phraser(trigram)
  return bigram_mod, trigram_mod


#  Further Cleaning and setting the position tags to nouns, adjectives, verbs and adverbss
def process_words(texts, bigram_mod, trigram_mod, stop_words=stop, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    # remove stopwords once more after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
    return texts_out


# Wordcloud of generated Topics
def plot_wordcloud(filename, lda_model):
  cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
  cloud = WordCloud(stopwords=stop,
                    background_color='white',
                    width=2500,
                    height=1800,
                    max_words=10,
                    colormap='tab10',
                    color_func=lambda *args, **kwargs: cols[i],
                    prefer_horizontal=1.0)
  print(f'10 Topics for {filename}')
  # lda_model = lda.LdaModel.load('LDA '+filename+'.model')
  topics = lda_model.show_topics(formatted=False)
  fig, axes = plt.subplots(5, 2, figsize=(30,30), sharex=True, sharey=True)

  for i, ax in enumerate(axes.flatten()):
      fig.add_subplot(ax)
      topic_words = dict(topics[i][1])
      cloud.generate_from_frequencies(topic_words, max_font_size=300)
      plt.gca().imshow(cloud)
      plt.gca().set_title('Topic ' + str(i+1), fontdict=dict(size=16))
      plt.gca().axis('off')
  plt.subplots_adjust(wspace=0, hspace=0)
  plt.axis('off')
  plt.margins(x=0, y=0)
  plt.tight_layout()
  plt.title(filename+' Topic Distribution')


#  Printing the Topic generated from the document with Positive & Negative Reviews
# Create Dictionary of Positive Comments
def LDA_generator(content, data_ready, random_state, update_every, filename):
  id2word = corpora.Dictionary(data_ready)

  # Create Corpus: Term Document Frequency
  corpus = [id2word.doc2bow(text) for text in data_ready]

  # Build LDA model
  lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=10, # 10
                                            random_state=random_state,
                                            update_every=update_every,
                                            chunksize=10, # 10
                                            passes=10,
                                            alpha='symmetric',
                                            iterations=500,
                                            per_word_topics=True)
 
  print(f'{filename} Printing now!')
  pprint(lda_model.print_topics())
  print('Done')
  fpath = 'model/LDAmodel'+filename+'.model'
  print('Saving the model at {fpath}')
  lda_model.save(fpath)
  print('Saved')
  print(f'Plotting the Wordcloud now..')
  plot_wordcloud(filename=filename,lda_model=lda_model)
  return lda_model

if __name__ == "__main__":
  for val, title in zip(['positives', 'negatives'], ['Postive Reviews', 'Negative Reviews']):
    print(f'Working on {title}')
    # Initial cleaning - STEP A
    print(f'Creating the clean sentences for {title}')
    data_words = create_data_words(val)
    print('Done!')
    # Generating the Gram model - STEP B
    print(f'Generating the Gram models')
    bigram_model, trigram_model = generate_gram_model(data_words)
    print('Done!')
    # Processing the reviews - STEP C
    print(f'Processing the clean sentences for {title}')
    data_ready = process_words(data_words, bigram_mod=bigram_model, trigram_mod=trigram_model)
    print('Done!')
    print(f'Creating the topics for {title}')
    LDA_generator(content=df[val], data_ready=data_ready, random_state=100, update_every=1, filename=title)
    print('Done!')
    print('*'*80)
  