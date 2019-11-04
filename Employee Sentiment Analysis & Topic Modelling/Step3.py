from Step2 import *

# Get topic weights and dominant topics ------------
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
import numpy as np
import pandas as pd

for val, title in zip(['positives', 'negatives'], ['Positive Reviews', 'Negative Reviews']):
    if val=='negatives':
        print('*'*80)
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
    # Loading the model - STEP D
    print(f'Loading the model for {title}')
    print('Done!')
    print(f'Creating the topics for {title}')
    lda_model = LDA_generator(content=df[val], data_ready=data_ready, random_state=100, update_every=1, filename=title)
    print('Done!')
    id2word = corpora.Dictionary(data_ready)
    corpus = [id2word.doc2bow(text) for text in data_ready]
    # Get topic weights
    print('Getting the topic weights')
    topic_weights = []
    for i, row_list in enumerate(lda_model[corpus]):
          topic_weights.append([w for i, w in row_list[0]])
    print('Done!')
    # Array of topic weights    
    arr = pd.DataFrame(topic_weights).fillna(0).values

    # Keep the well separated points (optional)
    arr = arr[np.amax(arr, axis=1) > 0.35]

    # Dominant topic number in each doc
    topic_num = np.argmax(arr, axis=1)

    # tSNE Dimension Reduction
    print('Starting the t-SNE model now..')
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(arr)
    print('t-SNE Parameters done!')

    # Plot the Topic Clusters using Bokeh
    output_notebook()
    n_topics = 10
    print(f'Plotting the Clustering model for {n_topics} topics')
    mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
    plot = figure(title="t-SNE Clustering of {} LDA Topics - {}".format(n_topics, title), 
                  plot_width=900, plot_height=700)
    plot.title.text_font_size = '15pt'
    plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
    # Increase readability
    plot.xaxis.axis_label_text_font_size = "20pt"
    plot.yaxis.axis_label_text_font_size = "20pt"
    plot.xgrid.grid_line_color = None
    plot.ygrid.grid_line_color = None
    show(plot)
