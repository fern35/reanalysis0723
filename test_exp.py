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

NL vars

"""
## lampe_Type: {}
## int_ElemDefaut: {'cover':['Crosse','Vasque','Enveloppe exterieure','Support','Coffret'],
## 'electricity':['Armorceur','Platine','Lampe','Câbles','Appareillage','Ballast','Protection électrique'],
## 'else':['NA','Luminaire','Armoire départ','Horloge','Alimentation générale']}


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
# vect_int_Defaut,vectorizer_int_Defaut = processor.var_vectorize(data=data_Int['int_Defaut'],vectorizer=TfidfVectorizer(max_features=100,strip_accents='unicode',stop_words=stopwords.words('french')))
# vect_pan_Defaut,vectorizer_pan_Defaut = processor.var_vectorize(data=data_Int['pan_Defaut'],vectorizer=TfidfVectorizer(max_features=100,strip_accents='unicode',stop_words=stopwords.words('french')))
# vect_int_ElemDefaut,vectorizer_int_ElemDefaut = processor.var_vectorize(data=data_Int['int_ElemDefaut'],vectorizer=TfidfVectorizer(max_features=100,strip_accents='unicode',stop_words=stopwords.words('french')))
# vect_int_TypeInt,vectorizer_int_TypeInt = processor.var_vectorize(data=data_Int['int_TypeInt'],vectorizer=TfidfVectorizer(max_features=100,strip_accents='unicode',stop_words=stopwords.words('french')))

# analyzer.get_tfvect_feature(vectorizer=vectorizer_pan_Defaut,varname='pan_Defaut')
# analyzer.get_tfvect_feature(vectorizer=vectorizer_int_Defaut,varname='int_Defaut')
# analyzer.get_tfvect_feature(vectorizer=vectorizer_int_ElemDefaut,varname='int_ElemDefaut')
# analyzer.get_tfvect_feature(vectorizer=vectorizer_int_TypeInt,varname='int_TypeInt')

### kmeans - cluster the doc-term matrix and see the centroids for each cluster
n_max = 20
# analyzer.ana_gapstat_kmeans(n_cluster_max=n_max,data_matrix=vect_pan_Defaut,titlename='kmeans_gapstat_pan_Defaut')
# analyzer.ana_gapstat_kmeans(n_cluster_max=n_max,data_matrix=vect_int_Defaut,titlename='kmeans_gapstat_int_Defaut')
# analyzer.ana_gapstat_kmeans(n_cluster_max=n_max,data_matrix=vect_int_TypeInt,titlename='kmeans_gapstat_int_TypeInt')
# analyzer.ana_gapstat_kmeans(n_cluster_max=n_max,data_matrix=vect_int_ElemDefaut,titlename='kmeans_gapstat_int_ElemDefaut')

### from the gap stat, we found that it is not easy for defining k for kmeans, because gap stat always increases
#### group the vars: int_ElemDefaut, int_TypeInt, int_Defaut, pan_Defaut
## int_ElemDefaut: just need to recount the categories ==> define a function, auto extrat the categories and return the encodeed

### create a function for retrieving the categories
# df_split_cat = analyzer.split_cat(data=data_Int,Var_lst=['int_ElemDefaut','int_TypeInt', 'int_TypeEqt',
#                                         'pan_TypeEqt', 'pan_Defaut', 'int_Defaut','int_Constat'])
#
# saver.save_excel(data=df_split_cat,filename='split_cat_Int')

### for int_TypeTnt, pan_Defaut, int_Defaut, I just take them as NL variables
### NL vars : int_Commentaire, pan_Commentaire
## check the NAN situation of BDD Intervention
# analyzer.gen_NAN_excel(data=data_Int,titlename='Int')

# vect_int_Commentaire,vectorizer_int_Commentaire = processor.var_vectorize(data=data_Int['int_Commentaire'],vectorizer=TfidfVectorizer(max_features=None,strip_accents='unicode',stop_words=stopwords.words('french')))
# vect_pan_Commentaire,vectorizer_pan_Commentaire = processor.var_vectorize(data=data_Int['pan_Commentaire'],vectorizer=TfidfVectorizer(max_features=None,strip_accents='unicode',stop_words=stopwords.words('french')))
#
# analyzer.get_tfvect_feature(vectorizer=vectorizer_int_Commentaire,varname='int_Commentaire')
# analyzer.get_tfvect_feature(vectorizer=vectorizer_pan_Commentaire,varname='pan_Commentaire')

### prob: NL var : too many special code? cannot understand ==> check if we can translate the code,
### if not, check the percentage of useful comment and decide if  we dump the variable

"""merge 3 BDD and define the target label
calculate the count of intervention for every eqt， 
decide the number of last interventions
"""

# analyzer.get_distr_inter(data=data_Int,Var_lst=['int_CodeEqt','int_Ville'],filename='obser_eqt_Int')
