from modules.loader import Loader
from modules.saver import Saver
from modules.analyser import Analyzer
from modules.cleaner import Cleaner
from modules.plotter import Plotter
from modules.modeler import Modeler
from modules.clustering import Cluster
from utils.constants import Armoire_CLUSTER_COMP, PL_CLUSTER_COMP
from modules.preprocessor import Processor
from modules.preprocessor import Processor
import datetime as dt
import seaborn as sns
import os
import pandas as pd
from utils.constants import Armoire_MERGE,Int_MERGE,PL_MERGE
import sklearn.feature_selection
from sklearn.feature_selection import f_regression, mutual_info_regression
from scipy.cluster.hierarchy import linkage


# the date of saving the data
date_str='0723'
analyzer = Analyzer(datestr=date_str)
cleaner = Cleaner()
loader = Loader(datestr=date_str)
saver = Saver(datestr=date_str)
processor = Processor(datestr=date_str)
plotter = Plotter(datestr=date_str)
cluster = Cluster(datestr=date_str)
modeler = Modeler(datestr=date_str)

data_downloadtime = dt.datetime(2018, 5, 15, 0, 0, 0, 0)
data_starttime = dt.datetime(2013, 1, 1, 0, 0, 0, 0)
day_difference = (data_downloadtime - data_starttime).days

CURRENT_TIME_AP = '2018-05-15'
CURRENT_TIME_INT = '2018_05_15'
Intfilename_lst = ["BDDExportInterventions-{} du 01_01_2013 au 15_05_2018.xlsx".format(CURRENT_TIME_INT)]

"""Attention: for this project, I dupmp the data of BOGOR
"""
"""
Merge the observations of all the cities and save to excel, this will be done by the api later
The related files are stored in data_save/excel/
"""
# for ville in VILLE_NAME:
#     # rename the dataframe,remove redundant info and save
#     data_Arm = loader.load_ArmPL(foldername=ville,filename="BDDExport_ArmoireBt_{}_{}.xlsx".format(ville,CURRENT_TIME_AP), NAME_LIST=Armoire_NAME)
#     data_PL = loader.load_ArmPL(foldername=ville,filename="BDDExport_PointLumineux_{}_{}.xlsx".format(ville,CURRENT_TIME_AP), NAME_LIST=PL_NAME)
#     data_Int = loader.load_Intervention(foldername=ville,filename_lst=Intfilename_lst, NAME_LIST=Int_NAME)
#
#     data_Arm = cleaner.rv_dupRow(data_Arm)
#     data_Ar = cleaner.rep_dur(data_Arm, Var_lst=Armoire_TIME, currtime=dt.datetime(2018, 5, 15, 0, 0, 0, 0))
#     data_PL = cleaner.rv_dupRow(data_PL)
#     data_PL = cleaner.rep_dur(data_PL, Var_lst=PL_TIME, currtime=dt.datetime(2018, 5, 15, 0, 0, 0, 0))
#     data_Int = cleaner.rv_dupRow(data_Int)
#     data_Int = cleaner.rep_dur(data_Int, Var_lst=Int_TIME, currtime=dt.datetime(2018, 5, 15, 0, 0, 0, 0))
#
#     saver.save_excel(data_Arm,foldername='Armoire',filename='Armoire_{}'.format(ville))
#     saver.save_excel(data_PL,foldername='PL',filename='PL_{}'.format(ville))
#     saver.save_excel(data_Int,foldername='Int',filename='Int_{}'.format(ville))


# data_Armoire = processor.merge_file(foldername='Armoire', villelst=VILLE_NAME, add_region=True)
# saver.save_excel(data_Armoire, foldername='Armoire',filename='Armoire_allcities')
#
# data_PL = processor.merge_file(foldername='PL', villelst=VILLE_NAME, add_region=True)
# saver.save_excel(data_PL, foldername='PL', filename='PL_allcities')
#
# data_Int = processor.merge_file(foldername='Int', villelst=VILLE_NAME, add_region=True)
# saver.save_excel(data_Int, foldername='Int', filename='Int_allcities')
#

"""
Merge Intervantion with Armoire and PL, 
"""
# data_Armoire = loader.load_excel(foldername='Armoire',filename='Armoire_allcities')
# data_Int = loader.load_excel(foldername='Int',filename='Int_allcities')
# data_PL = loader.load_excel(foldername='PL',filename='PL_allcities')
### simply merge
### use region and code for merge, because the var 'Ville' could be empty
### Armoire: 'eq_Code','region'
# data_select_Armoire = data_Armoire[['eq_Code','region']+Armoire_MERGE]
# data_select_Int_Arm = data_Int[['pan_CodeEqt','region']+Int_MERGE]
# data_merge_select_Arm = processor.merge_EqtInt(data_Eqt=data_select_Armoire,data_Int=data_select_Int_Arm)
# saver.save_excel(data=data_merge_select_Arm,filename='merge_ArmoireInt')
# ### PL:
# data_select_PL = data_PL[['eq_Code','region']+PL_MERGE]
# data_select_Int_PL = data_Int[['pan_CodeEqt','region']+Int_MERGE]
# data_merge_select_PL = processor.merge_EqtInt(data_Eqt=data_select_PL,data_Int=data_select_Int_PL)
# saver.save_excel(data=data_merge_select_PL,filename='merge_PLInt')

"""
construct target label, encode categorical vars, create functions for dealing with multi encoding,
standardize numerical vars, 
create a new df which only includes the interesting, transformed variables ==> feature dataframe
"""
"""
create new variables: delai_int(replace int_Fin),  int_Fin0-pan_DateSignal1, int_Fin1-pan_DateSignal2
Armoire:
"""
### create new variables: delai_int(replace int_Fin),  int_Fin0-pan_DateSignal1, int_Fin1-pan_DateSignal2
### Armoire:

## ADD NEW VARS
# data_merge_select_Arm = loader.load_excel(filename='merge_ArmoireInt')
# new_merge_ArmoireInt = processor.create_newvar(data_merge=data_merge_select_Arm)
# saver.save_excel(data=new_merge_ArmoireInt,filename='merge_ArmInt_addnewvar',foldername='Encode/Armoire')
# new_ArmInt_drop = new_merge_ArmoireInt.dropna(subset=['PanneDelai_1'])
# saver.save_excel(data=new_ArmInt_drop,filename='merge_ArmInt_dropna_PanneDelai_1',foldername='Encode/Armoire')

## encode
## the variables to encode:   (label is 'PanneDelai_1')
# MERGE_ENCODE_Arm = ['arm_NoLampe','eq_Vetuste','eq_Commentaire',
# 'pan_Commentaire_1','pan_Defaut_1','pan_Solde_1', 'int_ElemDefaut_1','int_Commentaire_1', 'DelaiInt_1',
# 'pan_Commentaire_2','pan_Defaut_2','pan_Solde_2', 'int_ElemDefaut_2','int_Commentaire_2', 'DelaiInt_2', 'PanneDelai_2',
# 'PanneDelai_1']
# new_ArmInt_drop = loader.load_excel(filename='merge_ArmInt_dropna_PanneDelai_1',foldername='Encode/Armoire')
# ## pick the variables
# new_ArmInt_encode_pickvar = new_ArmInt_drop[MERGE_ENCODE_Arm]
# ## encode numerical vars
# ArmInt_encode = processor.num_encode(data=new_ArmInt_encode_pickvar,var='arm_NoLampe',proper_range=(0,300))
# ArmInt_encode = processor.num_encode(data=ArmInt_encode,var='DelaiInt_1',proper_range=(0,1500))
# ArmInt_encode = processor.num_encode(data=ArmInt_encode,var='DelaiInt_2',proper_range=(0,1500))
# ArmInt_encode = processor.num_encode(data=ArmInt_encode,var='PanneDelai_1',proper_range=(0,1500))
# ArmInt_encode = processor.num_encode(data=ArmInt_encode,var='PanneDelai_2',proper_range=(0,1500))
# saver.save_excel(data=ArmInt_encode, filename='ArmInt_encode_num',foldername='Encode/Armoire')
# ## encode categorical vars
# ArmInt_encode = loader.load_excel(filename='ArmInt_encode_num',foldername='Encode/Armoire')
# dict_encode_eq_Vetuste_Armoire, ArmInt_encode = processor.cat_encode(data=ArmInt_encode,var='eq_Vetuste',regroup_dict=None)
# saver.save_pickle(data=dict_encode_eq_Vetuste_Armoire,filename='dict_encode_eq_Vetuste_Armoire')
# dict_encode_pan_Solde_1_Armoire, ArmInt_encode = processor.cat_encode(data=ArmInt_encode, var='pan_Solde_1',
#                                                             regroup_dict={'Solde':['Soldé'],
#                                                                           'Nonsolde':['Mise en provisoire','Mise en sécurité','Problème non résolu','Mise en attente'],
#                                                                           'Encours':['En cours']
#                                                                           # ,'Else':['NA']
#                                                                           })
# saver.save_pickle(data=dict_encode_pan_Solde_1_Armoire,filename='dict_encode_pan_Solde_1_Armoire')
# dict_encode_pan_Solde_2_Armoire, ArmInt_encode = processor.cat_encode(data=ArmInt_encode, var='pan_Solde_2',
#                                                             regroup_dict={'Solde':['Soldé'],
#                                                                           'Nonsolde':['Mise en provisoire','Mise en sécurité','Problème non résolu','Mise en attente']
#                                                                           # ,'Encours':['En cours']
#                                                                           ,'Else':['NA']
#                                                                           })
# saver.save_pickle(data=dict_encode_pan_Solde_2_Armoire,filename='dict_encode_pan_Solde_2_Armoire')
# ArmInt_encode = processor.cat_multi_encode(data=ArmInt_encode,
#                                            var='int_ElemDefaut_1',
#                                            regroup_dict={'Nonelectric':['Crosse','Vasque','Support','Enveloppe exterieure','Coffret','Trappe'],
#                                                          'Electric':['Câbles','Amorceur','Armoire départ','Ballast','Platine','Protection électrique','Lampe','Appareillage'],
#                                                          'Else':['NA','Alimentation générale','Luminaire','Horloge']})
# ArmInt_encode = processor.cat_multi_encode(data=ArmInt_encode,
#                                            var='int_ElemDefaut_2',
#                                            regroup_dict={'Nonelectric':['Crosse','Vasque','Support','Enveloppe exterieure','Coffret','Trappe'],
#                                                          'Electric':['Câbles','Amorceur','Armoire départ','Ballast','Platine','Protection électrique','Lampe','Appareillage'],
#                                                          'Else':['NA','Alimentation générale','Luminaire','Horloge']})
# saver.save_excel(data=ArmInt_encode, filename='ArmInt_encode_cat',foldername='Encode/Armoire')
#
# ## encode NL vars
# ArmInt_encode = loader.load_excel(filename='ArmInt_encode_cat',foldername='Encode/Armoire')
# ArmInt_encode = processor.NL_encode(data=ArmInt_encode,var='eq_Commentaire',max_feature=50)
# ArmInt_encode = processor.NL_encode(data=ArmInt_encode,var='pan_Commentaire_1',max_feature=50)
# ArmInt_encode = processor.NL_encode(data=ArmInt_encode,var='pan_Commentaire_2',max_feature=50)
# ArmInt_encode = processor.NL_encode(data=ArmInt_encode,var='pan_Defaut_1',max_feature=50)
# ArmInt_encode = processor.NL_encode(data=ArmInt_encode,var='pan_Defaut_2',max_feature=50)
# ArmInt_encode = processor.NL_encode(data=ArmInt_encode,var='int_Commentaire_1',max_feature=50)
# ArmInt_encode = processor.NL_encode(data=ArmInt_encode,var='int_Commentaire_2',max_feature=50)
# saver.save_excel(data=ArmInt_encode, filename='ArmInt_encode_NL',foldername='Encode/Armoire')
# #
# # ## next we will score the variables , but not considering NL vars
# # ArmInt_encode = loader.load_excel(filename='ArmInt_encode_NL',foldername='Encode')
# score_lst_Arm = ['arm_NoLampe','eq_Vetuste',
# 'pan_Solde_1', 'int_ElemDefaut_1_Electric', 'int_ElemDefaut_1_Nonelectric','int_ElemDefaut_1_Else','DelaiInt_1',
# 'pan_Solde_2', 'int_ElemDefaut_2_Electric', 'int_ElemDefaut_2_Nonelectric','int_ElemDefaut_2_Else','DelaiInt_2', 'PanneDelai_2']
# discrete_lst = [False,True,
#                 True,True,True,True,False,
#                 True, True, True, True, False,False]
# mi_MI = mutual_info_regression(ArmInt_encode[score_lst_Arm], ArmInt_encode['PanneDelai_1'],discrete_features=discrete_lst)
# df_mi_MI = pd.DataFrame(columns=['feature','score'])
# df_mi_MI['feature'] = score_lst_Arm
# df_mi_MI['score'] = mi_MI
# saver.save_excel(data=df_mi_MI,filename='feature_scoreMI_Armoire',foldername='Encode/Armoire')
#
# mi_F, pvalue_F = f_regression(ArmInt_encode[score_lst], ArmInt_encode['PanneDelai_1'],center=True)
# print(mi_F)
# df_mi_F = pd.DataFrame(columns=['feature','score','pvalue'])
# df_mi_F['feature'] = score_lst_Arm
# df_mi_F['score'] = mi_F
# df_mi_F['pvalue'] = pvalue_F
# saver.save_excel(data=df_mi_F,filename='feature_scoreF_Armoire',foldername='Encode/Armoire')

"""
create new variables: delai_int(replace int_Fin),  int_Fin0-pan_DateSignal1, int_Fin1-pan_DateSignal2
PL
"""
## ADD NEW VARS
# data_merge_select_PL = loader.load_excel(filename='merge_PLInt')
# new_merge_PLInt = processor.create_newvar(data_merge=data_merge_select_PL)
# saver.save_excel(data=new_merge_PLInt,filename='merge_PLInt_addnewvar',foldername='Encode/Int')

### drop the observations without the value of 'PanneDelai_1'
# new_PLInt_drop = new_merge_PLInt.dropna(subset=['PanneDelai_1'])
# saver.save_excel(data=new_PLInt_drop,filename='merge_PLInt_dropna_PanneDelai_1',foldername='Encode/PL')

## encode
## the variables to encode:   (label is 'PanneDelai_1')
# MERGE_ENCODE_PL = ['pl_Reseau','pl_NoLanterne','lan_Vetuste','lampe_Puissance','lampe_Type',
# 'pan_Commentaire_1','pan_Defaut_1','pan_Solde_1', 'int_ElemDefaut_1','int_Commentaire_1', 'DelaiInt_1',
# 'pan_Commentaire_2','pan_Defaut_2','pan_Solde_2', 'int_ElemDefaut_2','int_Commentaire_2', 'DelaiInt_2', 'PanneDelai_2',
# 'PanneDelai_1']
# new_PLInt_drop = loader.load_excel(filename='merge_PLInt_dropna_PanneDelai_1',foldername='Encode/PL')
#
# ## pick the variables
# new_PLInt_encode_pickvar = new_PLInt_drop[MERGE_ENCODE_PL]
# ## encode numerical vars
# PL_encode = processor.num_encode(data=new_PLInt_encode_pickvar,var='pl_NoLanterne',proper_range=None)
# PL_encode = processor.num_encode(data=PL_encode,var='lampe_Puissance',proper_range=(0,2000))
# PL_encode = processor.num_encode(data=PL_encode,var='DelaiInt_1',proper_range=(0,1500))
# PL_encode = processor.num_encode(data=PL_encode,var='DelaiInt_2',proper_range=(0,1500))
# PL_encode = processor.num_encode(data=PL_encode,var='PanneDelai_1',proper_range=(0,1500))
# PL_encode = processor.num_encode(data=PL_encode,var='PanneDelai_2',proper_range=(0,1500))
# saver.save_excel(data=PL_encode, filename='PLInt_encode_num',foldername='Encode/PL')

# ## encode categorical vars
# PL_encode = loader.load_excel(filename='PLInt_encode_num',foldername='Encode/PL')
# dict_encode_pl_Reseau_PL, PL_encode = processor.cat_encode(data=PL_encode,var='pl_Reseau',regroup_dict=None)
# saver.save_pickle(data=dict_encode_pl_Reseau_PL,filename='dict_encode_pl_Reseau_PL')
# dict_encode_lan_Vetuste_PL, PL_encode = processor.cat_encode(data=PL_encode,var='lan_Vetuste',regroup_dict=None)
# saver.save_pickle(data=dict_encode_lan_Vetuste_PL,filename='dict_encode_lan_Vetuste_PL')
# dict_encode_lampe_Type_PL, PL_encode = processor.cat_encode(data=PL_encode,var='lampe_Type',
#                                                          regroup_dict={'SP':['SBP','SHP'], 'M':['IM','ML','COSMOWHITE'],
#                                                                        'F':['BF','Fluo'],'Else':['INC','HAL','LED','NA']})
# saver.save_pickle(data=dict_encode_lampe_Type_PL,filename='dict_encode_lampe_Type_PL')
#
# dict_encode_pan_Solde_1_PL, PL_encode = processor.cat_encode(data=PL_encode, var='pan_Solde_1',
#                                                             regroup_dict={'Solde':['Soldé'],
#                                                                           'Nonsolde':['Mise en provisoire','Mise en sécurité','Problème non résolu','Mise en attente'],
#                                                                           'Encours':['En cours']
#                                                                           # ,'Else':['NA']
#                                                                           })
# saver.save_pickle(data=dict_encode_pan_Solde_1_PL,filename='dict_encode_pan_Solde_1_PL')
# dict_encode_pan_Solde_2_PL, PL_encode = processor.cat_encode(data=PL_encode, var='pan_Solde_2',
#                                                             regroup_dict={'Solde':['Soldé'],
#                                                                           'Nonsolde':['Mise en provisoire','Mise en sécurité','Problème non résolu','Mise en attente'],
#                                                                           'Encours':['En cours'],
#                                                                           'Else':['NA']})
# saver.save_pickle(data=dict_encode_pan_Solde_2_PL,filename='dict_encode_pan_Solde_2_PL')
# PL_encode = processor.cat_multi_encode(data=PL_encode,
#                                            var='int_ElemDefaut_1',
#                                            regroup_dict={'Nonelectric':['Crosse','Vasque','Support','Enveloppe exterieure','Coffret','Trappe'],
#                                                          'Electric':['Câbles','Amorceur','Armoire départ','Ballast','Platine','Protection électrique','Lampe','Appareillage'],
#                                                          'Else':['NA','Alimentation générale','Luminaire','Horloge']})
# PL_encode = processor.cat_multi_encode(data=PL_encode,
#                                            var='int_ElemDefaut_2',
#                                            regroup_dict={'Nonelectric':['Crosse','Vasque','Support','Enveloppe exterieure','Coffret','Trappe'],
#                                                          'Electric':['Câbles','Amorceur','Armoire départ','Ballast','Platine','Protection électrique','Lampe','Appareillage'],
#                                                          'Else':['NA','Alimentation générale','Luminaire','Horloge']})
# saver.save_excel(data=PL_encode, filename='PL_encode_cat',foldername='Encode/PL')
#
#
# # encode NL vars
# PL_encode = loader.load_excel(filename='PL_encode_cat',foldername='Encode/PL')
# PL_encode = processor.NL_encode(data=PL_encode,var='pan_Commentaire_1',max_feature=50)
# PL_encode = processor.NL_encode(data=PL_encode,var='pan_Commentaire_2',max_feature=50)
# PL_encode = processor.NL_encode(data=PL_encode,var='pan_Defaut_1',max_feature=50)
# PL_encode = processor.NL_encode(data=PL_encode,var='pan_Defaut_2',max_feature=50)
# PL_encode = processor.NL_encode(data=PL_encode,var='int_Commentaire_1',max_feature=50)
# PL_encode = processor.NL_encode(data=PL_encode,var='int_Commentaire_2',max_feature=50)
# saver.save_excel(data=PL_encode, filename='PL_encode_NL',foldername='Encode/PL')

# ## next we will score the variables , but not considering NL vars
# PL_encode = loader.load_excel(filename='PL_encode_NL',foldername='Encode/PL')
PL_score_lst = ['pl_Reseau','pl_NoLanterne','lan_Vetuste','lampe_Puissance','lampe_Type',
'pan_Solde_1', 'int_ElemDefaut_1_Electric', 'int_ElemDefaut_1_Nonelectric','int_ElemDefaut_1_Else','DelaiInt_1',
'pan_Solde_2', 'int_ElemDefaut_2_Electric', 'int_ElemDefaut_2_Nonelectric','int_ElemDefaut_2_Else','DelaiInt_2', 'PanneDelai_2']
# discrete_lst = [True,True,True,False,True,
#                 True,True,True,True,False,
#                 True, True, True, True, False,False]
# mi_MI = mutual_info_regression(PL_encode[PL_score_lst], PL_encode['PanneDelai_1'],discrete_features=discrete_lst)
# df_mi_MI = pd.DataFrame(columns=['feature','score'])
# df_mi_MI['feature'] = PL_score_lst
# df_mi_MI['score'] = mi_MI
# saver.save_excel(data=df_mi_MI,filename='feature_scoreMI_PL',foldername='Encode/PL')
#
# mi_F, pvalue_F = f_regression(PL_encode[PL_score_lst], PL_encode['PanneDelai_1'],center=True)
# print(mi_F)
# df_mi_F = pd.DataFrame(columns=['feature','score','pvalue'])
# df_mi_F['feature'] = PL_score_lst
# df_mi_F['score'] = mi_F
# df_mi_F['pvalue'] = pvalue_F
# saver.save_excel(data=df_mi_F,filename='feature_scoreF_PL',foldername='Encode/PL')

"""
Clustering and visualization of the characteristics of the clusters
"""
""" Armoire
"""
# ## normalize numerical vars and turn categorical variables to one hot
# ArmInt_encode = loader.load_excel(filename='ArmInt_encode_NL',foldername='Encode/Armoire')
# ArmInt_cluster = processor.turn_onehot(data=ArmInt_encode,var='eq_Vetuste',namedict=dict_encode_eq_Vetuste_Armoire)
# ArmInt_cluster = processor.turn_onehot(data=ArmInt_cluster,var='pan_Solde_1',namedict=dict_encode_pan_Solde_1_Armoire)
# ArmInt_cluster = processor.turn_onehot(data=ArmInt_cluster,var='pan_Solde_2',namedict=dict_encode_pan_Solde_2_Armoire)
#
# ArmInt_cluster = processor.scale_num(data=ArmInt_cluster,var='arm_NoLampe')
# ArmInt_cluster = processor.scale_num(data=ArmInt_cluster,var='DelaiInt_1')
# ArmInt_cluster = processor.scale_num(data=ArmInt_cluster,var='DelaiInt_2')
# ArmInt_cluster = processor.scale_num(data=ArmInt_cluster,var='PanneDelai_1')
# ArmInt_cluster = processor.scale_num(data=ArmInt_cluster,var='PanneDelai_2')
#
# saver.save_excel(data=ArmInt_cluster,filename='ArmInt_cluster',foldername='Cluster')

# ArmInt_cluster = loader.load_excel(filename='ArmInt_cluster',foldername='Cluster')
# ArmInt_cluster.drop(['PanneDelai_1'], axis=1,inplace=True)
# hcluster_Armoire = linkage(ArmInt_cluster, 'ward')
# plotter.plot_dendro(hcluster_Armoire,'clustering','Dendrogram_Armoire')

## generaye files for cluster， and generate distribution of many vars
# new_ArmInt_data = loader.load_excel(filename='merge_ArmInt_dropna_PanneDelai_1',foldername='Encode/Armoire')
# new_ArmInt_data.reset_index(drop=True, inplace=True)
# ArmInt_cluster = loader.load_excel(filename='ArmInt_cluster',foldername='Cluster')
# hcluster_ArmInt = linkage(ArmInt_cluster, 'ward')
# saver.save_pickle(data=hcluster_ArmInt,filename='hcluster_ArmInt')
#
# n_cluster_lst = [3,4,5,8]
# for j in n_cluster_lst:
#     clsize_HC_ArmInt, df_HC_ArmInt = cluster.getcontent_HC(new_ArmInt_data, hcluster_ArmInt, j,
#                                                             colnames=list(new_ArmInt_data.columns))
#     saver.save_excel(df_HC_ArmInt, foldername='Cluster', filename='ArmInt_cluster_ncluster{}'.format(j))
# for j in n_cluster_lst:
#     df_Arm_cluster = loader.load_excel(foldername='Cluster', filename='ArmInt_cluster_ncluster{}'.format(j))
#     analyzer.comp_Var_cluster(data=df_Arm_cluster,group_dict=Armoire_CLUSTER_COMP,n_cluster=j,foldername='Armoire')
# df_Arm_cluster = loader.load_excel(foldername='Cluster', filename='PL_cluster_ncluster{}'.format(3))
# print(df_Arm_cluster.columns)
"""PL
"""
# PL_encode = loader.load_excel(filename='PL_encode_NL',foldername='Encode/PL')
# PL_cluster = processor.turn_onehot(data=PL_encode,var='pl_Reseau',namedict=dict_encode_pl_Reseau_PL)
# PL_cluster = processor.turn_onehot(data=PL_cluster,var='lan_Vetuste',namedict=dict_encode_lan_Vetuste_PL)
# PL_cluster =processor.turn_onehot(data=PL_cluster,var='lampe_Type',namedict=dict_encode_lampe_Type_PL)
# PL_cluster = processor.turn_onehot(data=PL_cluster,var='pan_Solde_1',namedict=dict_encode_pan_Solde_1_PL)
# PL_cluster = processor.turn_onehot(data=PL_cluster,var='pan_Solde_2',namedict=dict_encode_pan_Solde_2_PL)
#
# PL_cluster = processor.scale_num(data=PL_cluster,var='pl_NoLanterne')
# PL_cluster = processor.scale_num(data=PL_cluster,var='lampe_Puissance')
# PL_cluster = processor.scale_num(data=PL_cluster,var='DelaiInt_1')
# PL_cluster = processor.scale_num(data=PL_cluster,var='DelaiInt_2')
# PL_cluster = processor.scale_num(data=PL_cluster,var='PanneDelai_1')
# PL_cluster = processor.scale_num(data=PL_cluster,var='PanneDelai_2')
# saver.save_excel(data=PL_cluster,filename='PL_cluster',foldername='Cluster')
#
# PL_cluster = loader.load_excel(filename='PL_cluster',foldername='Cluster')
# PL_cluster.drop([PanneDelai_1], axis=1,inplace=True)
# hcluster_PL = linkage(PL_cluster, 'ward')
# saver.save_pickle(data=hcluster_PL,filename='hcluster_PL')
# plotter.plot_dendro(hcluster_PL,'clustering','Dendrogram_PL')

## generaye files for cluster， and generate distribution of many vars
# new_PLInt_data = loader.load_excel(filename='merge_PLInt_dropna_PanneDelai_1',foldername='Encode/PL')
# new_PLInt_data.reset_index(drop=True, inplace=True)
# PL_cluster = loader.load_excel(filename='PL_cluster',foldername='Cluster')
# hcluster_PL = linkage(PL_cluster, 'ward')
# saver.save_pickle(data=hcluster_PL,filename='hcluster_PL')
#
# n_cluster_lst = [3,4,5,8]
# for j in n_cluster_lst:
#     clsize_mixed_PL, df_mixed_PL = cluster.getcontent_HC(new_PLInt_data, hcluster_PL, j,
#                                                             colnames=list(new_PLInt_data.columns))
#     saver.save_excel(df_mixed_PL, foldername='Cluster', filename='PL_cluster_ncluster{}'.format(j))

### 'pan_Defaut' taken as NLP variable

# for j in n_cluster_lst:
#     df_PL_cluster = loader.load_excel(foldername='Cluster', filename='PL_cluster_ncluster{}'.format(j))
#     analyzer.comp_Var_cluster(data=df_PL_cluster,group_dict=PL_CLUSTER_COMP,n_cluster=j,foldername='PL')

"""visualization of clusters
"""


"""
correlation between variables
"""
# Var_lst_Armoire = ['pan_Solde_2_Solde', 'pan_Solde_2_Else', 'pan_Solde_2_Nonsolde',
#        'pan_Solde_1_Solde', 'pan_Solde_1_Encours', 'pan_Solde_1_Nonsolde',
#        'eq_Vetuste_Moyen', 'eq_Vetuste_HS', 'eq_Vetuste_Bon',
#                    'eq_Vetuste_Vétuste', 'arm_NoLampe', 'DelaiInt_2', 'PanneDelai_2',
#                    'PanneDelai_1','int_ElemDefaut_1_Electric', 'int_ElemDefaut_1_Else',
#        'int_ElemDefaut_1_Nonelectric', 'int_ElemDefaut_2_Electric',
#        'int_ElemDefaut_2_Else', 'int_ElemDefaut_2_Nonelectric'
#                    ]
# ArmInt_cluster = loader.load_excel(filename='ArmInt_cluster',foldername='Cluster')
# ArmInt_cluster.drop(['DelaiInt_1'], axis=1,inplace=True)
# # print(ArmInt_cluster.columns[150:])
# ArmInt_cluster = ArmInt_cluster[Var_lst_Armoire]
# cor_Arm = ArmInt_cluster.corr()
# cor_plot_Arm = sns.heatmap(cor_Arm, square = True).get_figure()
# cor_plot_Arm.savefig(os.path.join(saver.datasavedir,'img','clustering','correlation_Arm.jpg'))

#
# Var_lst_PL=['pan_Solde_2_Encours', 'pan_Solde_2_Nonsolde', 'pan_Solde_2_Solde',
#        'pan_Solde_2_Else', 'pan_Solde_1_Encours', 'pan_Solde_1_Nonsolde',
#        'pan_Solde_1_Solde', 'lampe_Type_M', 'lampe_Type_F', 'lampe_Type_Else',
#             'lampe_Type_SP', 'lan_Vetuste_Moyen', 'lan_Vetuste_Vétuste',
#             'lan_Vetuste_Bon', 'lan_Vetuste_NA', 'lan_Vetuste_HS',
#             'pl_Reseau_Souterrain', 'pl_Reseau_Aérien nu', 'pl_Reseau_Aérien mixte',
#             'pl_Reseau_PRC','pl_Reseau_Façade', 'pl_Reseau_NA', 'pl_Reseau_Aerien ECP Seul',
#        'pl_Reseau_Aérien', 'pl_NoLanterne', 'lampe_Puissance', 'DelaiInt_2',
#        'PanneDelai_2', 'PanneDelai_1','int_ElemDefaut_1_Electric', 'int_ElemDefaut_1_Else',
#             'int_ElemDefaut_1_Nonelectric', 'int_ElemDefaut_2_Electric', 'int_ElemDefaut_2_Else','int_ElemDefaut_2_Nonelectric'
#             ]
# PL_cluster = loader.load_excel(filename='PL_cluster',foldername='Cluster')
# PL_cluster.drop(['DelaiInt_1'], axis=1,inplace=True)
# PL_cluster = PL_cluster[Var_lst_PL]
# cor_PL = PL_cluster.corr()
# cor_plot_PL = sns.heatmap(cor_PL, square = True).get_figure()
# cor_plot_PL.savefig(os.path.join(saver.datasavedir,'img','clustering','correlation_PL.jpg'))


"""
random forest
"""
ArmInt_cluster = loader.load_excel(filename='ArmInt_cluster',foldername='Cluster')
ArmInt_num = loader.load_excel(filename='ArmInt_encode_num',foldername='Encode/Armoire')
ArmInt_num.reset_index(drop=True,inplace=True)
ArmInt_cluster[['PanneDelai_1','DelaiInt_1','PanneDelai_2','DelaiInt_2']] = ArmInt_num[['PanneDelai_1','DelaiInt_1','PanneDelai_2','DelaiInt_2']]
# print(ArmInt_cluster[['PanneDelai_1','DelaiInt_1','PanneDelai_2','DelaiInt_2']].head())
y = pd.DataFrame(ArmInt_cluster['PanneDelai_1']).values
ArmInt_cluster.drop(['PanneDelai_1'], axis=1,inplace=True)
X = ArmInt_cluster.values

# modeler.train_RandomForest(X=X,y=y,title='Armoire')
modeler.train_GradientBoosting(X=X,y=y,title='Armoire')