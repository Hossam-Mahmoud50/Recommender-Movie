# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 18:13:45 2018

@author: hosssam
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn import metrics
column_names = ['user_id', 'item_id', 'rating', 'timestamp'] 
df = pd.read_csv('file.tsv', sep='\t', names=column_names) 

# Check the head of the data 
df.head() 
# Check out all the movies and their respective IDs 
movie_titles = pd.read_csv('Movie_Id_Titles.csv') 
movie_titles.head() 
Movies_with_Rating = pd.merge(df, movie_titles, on='item_id') 
Movies_with_Rating.head() 
test=Movies_with_Rating.iloc[-1:90003 , :]

cloumn=['timestamp']
Movies_with_Rating=Movies_with_Rating.drop(cloumn,axis=1)
Movies_with_Rating.head()
Movies_with_Rating = Movies_with_Rating.dropna(axis=0,subset=['title'])
Movie_ratingCount = (Movies_with_Rating .
     groupby(by = ['title'])['rating'].
     count().
     reset_index().
     rename(columns = {'rating': 'totalRatingCount'})
     [['title', 'totalRatingCount']]
    )
Movie_ratingCount.head()

rating_with_totalRatingCount = Movies_with_Rating.merge(Movie_ratingCount, left_on = 'title', right_on ='title', how = 'left')
rating_with_totalRatingCount.head()
#compute the most 50 movie rate
popularity_threshold = 50
rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
rating_popular_book.head()

moviemat = Movies_with_Rating.pivot_table(index ='title', 
			columns ='user_id', values ='rating').fillna(0)

moviemat.head() 

Movie_ratingCount.sort_values('totalRatingCount', ascending = False).head(10) 
#Filter
user_rating_matrix = csr_matrix(moviemat.values)

from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(user_rating_matrix)

query_index = np.random.choice(moviemat.shape[0])
us=np.array(moviemat.iloc[query_index, :]).reshape(1, -1)
distances, indices = model_knn.kneighbors(us, n_neighbors = 6)
x_text=[]
for i in range(0, len(distances.flatten())):
   
    if i == 0:
    
        print('Recommendations for {0}:\n'.format(moviemat.index[query_index]))
    else:
        x_text.append(moviemat.index[indices.flatten()[i]])
        print('{0}: {1}, with distance of {2}:'.format(i, moviemat.index[indices.flatten()[i]], distances.flatten()[i]))
