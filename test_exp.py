# Check the MUT CAT vars and NL vars, find proper methods for processing and generate related constants
from modules.analyser import Analyzer
from modules.loader import Loader
from modules.saver import Saver
from modules.preprocessor import Processor
from utils.constants import VILLE_NAME, Armoire_PICK, Int_PICK, PL_PICK
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

date_str = '0723'
analyzer = Analyzer(datestr=date_str)
loader = Loader(datestr=date_str)
saver = Saver(datestr=date_str)
processor = Processor(datestr=date_str)
"""
MUL CAT vars
PL: lampe_Type
INT: pan_Solde, int_Solde, int_ElemDefaut, int_TypeTnt, int_TypeEqt, pan_TypeEqt, pan_Defaut, int_Defaut
"""
####### generate doc for analyzing the MUL CAT vars
# analyzer.comp_Var_cities(foldername='Armoire',villelst=VILLE_NAME,group_dict= Armoire_PICK)
# analyzer.comp_Var_cities(foldername='PL',villelst=VILLE_NAME,group_dict= PL_PICK)
# analyzer.comp_Var_cities(foldername='Int',villelst=VILLE_NAME,group_dict= Int_PICK)
####### Probs:
####### 1. int_Solde & pan_Solde, int_Defaut & pan_Defaut, int_TypeEqt & pan_TypeEqt should be the same
####### 2. merge the categories:
####### * pan_Solde, int_Solde completed
####### int_TypeEqt, pan_TypeEqt: mostly the same
####### pan_Defaut, int_Defaut: mostly the same
####### check all the categories of lampe_Type, int_ElemDefaut, int_TypeTnt, int_TypeEqt, pan_TypeEqt, pan_Defaut, int_Defaut
####### save to excel
data_Int = loader.load_excel(foldername='Int',filename='Int_allcities')
# data_PL = loader.load_excel(foldername='PL',filename='PL_allcities')
# df_PL = analyzer.gen_cat_var(data=data_PL,Var_lst=['lampe_Type'])
# df_Int = analyzer.gen_cat_var(data=data_Int,Var_lst=['int_ElemDefaut','int_TypeInt', 'int_TypeEqt',
#                                                        'pan_TypeEqt', 'pan_Defaut', 'int_Defaut'])
# df_mulcat = pd.concat([df_PL, df_Int], axis=1)
# saver.save_excel(data=df_mulcat,filename='cat_MULCAT')

####### Simple Strategy
####### PL: lampe_Type : conserve the top 6 categories
####### Int: pan_TypeEqt, int_TypeEqt: PL, Armoire, else
####### Int: int_Defaut, pan_Defaut ==> NL autoencoder, turned to vector, and clustering for visualization and ranks
####### Int: int_ElemDefaut ==> NL autoencoder, turned to vector, and clustering for visualization and ranks
####### Int: int_TypeInt ==> NL autoencoder, turned to vector, and clustering for visualization and ranks

####### NL encode
#### analysis
vect_int_Defaut,vectorizer_int_Defaut = processor.var_vectorize(data=data_Int['int_Defaut'],vectorizer=TfidfVectorizer(max_features=100,strip_accents='unicode',stop_words=stopwords.words('french')))
vect_pan_Defaut,vectorizer_pan_Defaut = processor.var_vectorize(data=data_Int['pan_Defaut'],vectorizer=TfidfVectorizer(max_features=100,strip_accents='unicode',stop_words=stopwords.words('french')))
analyzer.get_tfvect_feature(vectorizer=vectorizer_pan_Defaut,varname='pan_Defaut')
analyzer.get_tfvect_feature(vectorizer=vectorizer_int_Defaut,varname='int_Defaut')
### kmeans - cluster the doc-term matrix and see the centroids for each cluster
n_max = 20
analyzer.ana_gapstat_kmeans(n_cluster_max=20,data_matrix=vect_pan_Defaut,titlename='kmeans_gapstat_pan_Defaut')
analyzer.ana_gapstat_kmeans(n_cluster_max=20,data_matrix=vect_int_Defaut,titlename='kmeans_gapstat_int_Defaut')


