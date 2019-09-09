print('Loading spaCy and other packages... this may take a while.')

import sys, string
import numpy as np
import spacy

from stop_list import closed_class_stop_words as stop_words
from nltk.stem import SnowballStemmer as sbs
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine

stemmer = sbs('english')
nlp = spacy.load('en_core_web_lg')


mode = 'para'

#opening the document file 
documents_file = open(sys.argv[1], 'r')

if mode == 'sent':
    '''
    split text into sentence
    deprecated in favor of splitting text into paragraphs
    '''
    text = documents_file.read().replace('\n', ' ')

    sentences = []
    sentence_vectors = []
    
    doc = nlp(text)
    for sent in doc.sents:
        if sent.vector_norm > 0:
            sentences.append(sent.text)
            sentence_vectors.append(sent.vector)
            print(sent.text)
            print('-')

    vectors_matrix = np.asmatrix(sentence_vectors)    
else:
    #splitting text into paragraphs
    temp_paragraphs = documents_file.read().split('\n\n')
    
    sentences = []
    processed_sentences = []
    sentence_vectors = []
    
    #preprocessing
    for paragraph in temp_paragraphs:
        
        #deleting extra \n
        orig_p = paragraph.replace('\n', ' ')
        
        #deleting punctuations
        p = orig_p.translate(str.maketrans('', '', string.punctuation))
        
        #removing stop words
        #list of stop words is provided in class
        p_list = p.split()
        temp_p_list = []
        for word in p_list:
            if word not in stop_words:
                temp_p_list.append(word.lower())
        
        processed_p = ' '.join(temp_p_list)
        
        #convert text into vectors
        doc = nlp(processed_p)
        if doc.vector_norm > 0:
            processed_sentences.append(processed_p)
            sentences.append(orig_p)
            sentence_vectors.append(doc.vector)
            #print(doc.text)
            #print('-')
        
    vectors_matrix = np.asmatrix(sentence_vectors) 
    
    print('Number of paragraphs in input file: ' + str(len(sentences)))
    print('Text vectors matrix shape: ' + str(vectors_matrix.shape))
    
#setting max and min number of clusters
lower_bound = int(len(sentences) / 5)
upper_bound = int(len(sentences) / 2)

#performing K-Means clustering 
for c in range(lower_bound, upper_bound):
    kmeans = KMeans(n_clusters=c, random_state=0).fit(vectors_matrix)
    cluster_centers = kmeans.cluster_centers_.tolist()
    
    print('Number of clusters = Number of paragraphs in summary: ' + str(len(cluster_centers)))
    print('Cluster # for each paragraph: '+ str(kmeans.labels_))
    
    #getting the paragraph that is closest to the centroid of each cluster
    summary = []
    for center in cluster_centers:
        current_best = -1
        current_best_vectors = None
        current_best_index = None
        current_best_text = None

        for i in range(0, len(sentence_vectors)):
            para = sentence_vectors[i]
            similarity = 1 - cosine(para, center)

            if similarity > current_best:
                current_best = similarity
                current_best_vectors = para
                current_best_text = sentences[i]
                current_best_index = i

        temp_tuple = (current_best_text, current_best_index)
        summary.append(temp_tuple)

    #sorting the paragrapsh so that they are in the original order as the input file
    summary.sort(key=lambda tup: tup[1])  
    summary_text = ''
    for item in summary:
        summary_text += item[0] + '\n'

    #writing output to files
    out_name = 'output' + str(c) + '.txt'
    output_file = open(out_name, 'w')
    output_file.write(summary_text)
    output_file.close()

documents_file.close()

