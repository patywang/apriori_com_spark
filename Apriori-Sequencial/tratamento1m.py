import pandas as pd
from efficient_apriori import apriori

#carga dos arquivos

def carga2():
	col_data = ['user_id', 'movie_id', 'rating_user', 'timestamp_user']
	data = pd.read_csv("./ml-1m/ratings.dat", names=col_data,  delimiter="::", engine='python')
	col_item_data = ['movie_id', 'movie_title', 'categories']
	itemData = pd.read_csv("./ml-1m/movies.dat",  names=col_item_data, encoding="ISO-8859-1", delimiter="::", engine='python')
	dataset = pd.merge(data, itemData, on='movie_id', how='outer')
	return dataset

def tratamento(arquivo):

	treatment_doc = carga2()

	array = []
	tupla_name = ()
	for name,group in treatment_doc.groupby(['user_id']):
			tupla_name = tuple(group['movie_title'].values)
			array.append(tupla_name)
			
	
	#return array,listMovies, listUsers
	return array
