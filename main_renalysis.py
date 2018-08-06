from modules.loader import Loader
from modules.saver import Saver
from modules.analyser import Analyzer
from modules.cleaner import Cleaner
from modules.preprocessor import Processor
from modules.preprocessor import Processor
import datetime as dt
import pandas as pd
from utils.constants import Armoire_MERGE,Int_MERGE,PL_MERGE
import sklearn.feature_selection
from sklearn.feature_selection import f_regression, mutual_info_regression

# the date of saving the data
date_str='0723'
analyzer = Analyzer(datestr=date_str)
cleaner = Cleaner()
loader = Loader(datestr=date_str)
saver = Saver(datestr=date_str)
processor = Processor(datestr=date_str)

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
### create new variables: delai_int(replace int_Fin),  int_Fin0-pan_DateSignal1, int_Fin1-pan_DateSignal2
### Armoire:

## ADD NEW VARS
# data_merge_select_Arm = loader.load_excel(filename='merge_ArmoireInt')
# new_merge_ArmoireInt = processor.create_newvar(data_merge=data_merge_select_Arm)
# saver.save_excel(data=new_merge_ArmoireInt,filename='merge_ArmInt_addnewvar',foldername='Encode')
# new_ArmInt_drop = new_merge_ArmoireInt.dropna(subset=['PanneDelai_1'])
# saver.save_excel(data=new_ArmInt_drop,filename='merge_ArmInt_dropna_PanneDelai_1',foldername='Encode')

## encode
## the variables to encode:   (label is 'PanneDelai_1')
MERGE_ENCODE_Arm = ['arm_NoLampe','eq_Vetuste','eq_Commentaire',
'pan_Commentaire_1','pan_Defaut_1','pan_Solde_1', 'int_ElemDefaut_1','int_Commentaire_1', 'DelaiInt_1',
'pan_Commentaire_2','pan_Defaut_2','pan_Solde_2', 'int_ElemDefaut_2','int_Commentaire_2', 'DelaiInt_2', 'PanneDelai_2',
'PanneDelai_1']
new_ArmInt_drop = loader.load_excel(filename='merge_ArmInt_dropna_PanneDelai_1',foldername='Encode')
## pick the variables
new_ArmInt_encode_pickvar = new_ArmInt_drop[MERGE_ENCODE_Arm]
## encode numerical vars
ArmInt_encode = processor.num_encode(data=new_ArmInt_encode_pickvar,var='arm_NoLampe',proper_range=(0,300))
ArmInt_encode = processor.num_encode(data=ArmInt_encode,var='DelaiInt_1',proper_range=(0,1500))
ArmInt_encode = processor.num_encode(data=ArmInt_encode,var='DelaiInt_2',proper_range=(0,1500))
ArmInt_encode = processor.num_encode(data=ArmInt_encode,var='PanneDelai_1',proper_range=(0,1500))
ArmInt_encode = processor.num_encode(data=ArmInt_encode,var='PanneDelai_2',proper_range=(0,1500))
saver.save_excel(data=ArmInt_encode, filename='ArmInt_encode_num',foldername='Encode')
## encode categorical vars
ArmInt_encode = loader.load_excel(filename='ArmInt_encode_num',foldername='Encode')
dict_encode_eq_Vetuste, ArmInt_encode = processor.cat_encode(data=ArmInt_encode,var='eq_Vetuste',regroup_dict=None)
dict_encode_pan_Solde_1, ArmInt_encode = processor.cat_encode(data=ArmInt_encode, var='pan_Solde_1',
                                                            regroup_dict={'Solde':['Soldé'],
                                                                          'Nonsolde':['Mise en provisoire','Mise en sécurité','Problème non résolu','Mise en attente'],
                                                                          'Encours':['En cours'],
                                                                          'Else':['NA']})
print(dict_encode_pan_Solde_1)
dict_encode_pan_Solde_2, ArmInt_encode = processor.cat_encode(data=ArmInt_encode, var='pan_Solde_2',
                                                            regroup_dict={'Solde':['Soldé'],
                                                                          'Nonsolde':['Mise en provisoire','Mise en sécurité','Problème non résolu','Mise en attente'],
                                                                          'Encours':['En cours'],
                                                                          'Else':['NA']})
print(dict_encode_pan_Solde_2)
ArmInt_encode = processor.cat_multi_encode(data=ArmInt_encode,
                                           var='int_ElemDefaut_1',
                                           regroup_dict={'Nonelectric':['Crosse','Vasque','Support','Enveloppe exterieure','Coffret','Trappe'],
                                                         'Electric':['Câbles','Amorceur','Armoire départ','Ballast','Platine','Protection électrique','Lampe','Appareillage'],
                                                         'Else':['NA','Alimentation générale','Luminaire','Horloge']})
ArmInt_encode = processor.cat_multi_encode(data=ArmInt_encode,
                                           var='int_ElemDefaut_2',
                                           regroup_dict={'Nonelectric':['Crosse','Vasque','Support','Enveloppe exterieure','Coffret','Trappe'],
                                                         'Electric':['Câbles','Amorceur','Armoire départ','Ballast','Platine','Protection électrique','Lampe','Appareillage'],
                                                         'Else':['NA','Alimentation générale','Luminaire','Horloge']})
saver.save_excel(data=ArmInt_encode, filename='ArmInt_encode_cat',foldername='Encode')

## encode NL vars
ArmInt_encode = loader.load_excel(filename='ArmInt_encode_cat',foldername='Encode')
ArmInt_encode = processor.NL_encode(data=ArmInt_encode,var='eq_Commentaire',max_feature=50)
ArmInt_encode = processor.NL_encode(data=ArmInt_encode,var='pan_Commentaire_1',max_feature=50)
ArmInt_encode = processor.NL_encode(data=ArmInt_encode,var='pan_Commentaire_2',max_feature=50)
ArmInt_encode = processor.NL_encode(data=ArmInt_encode,var='pan_Defaut_1',max_feature=50)
ArmInt_encode = processor.NL_encode(data=ArmInt_encode,var='pan_Defaut_2',max_feature=50)
ArmInt_encode = processor.NL_encode(data=ArmInt_encode,var='int_Commentaire_1',max_feature=50)
ArmInt_encode = processor.NL_encode(data=ArmInt_encode,var='int_Commentaire_2',max_feature=50)
saver.save_excel(data=ArmInt_encode, filename='ArmInt_encode_NL',foldername='Encode')

## next we will score the variables , but not considering NL vars
ArmInt_encode = loader.load_excel(filename='ArmInt_encode_NL',foldername='Encode')
score_lst = ['arm_NoLampe','eq_Vetuste',
'pan_Solde_1', 'int_ElemDefaut_1_Electric', 'int_ElemDefaut_1_Nonelectric','int_ElemDefaut_1_Else','DelaiInt_1',
'pan_Solde_2', 'int_ElemDefaut_2_Electric', 'int_ElemDefaut_2_Nonelectric','int_ElemDefaut_2_Else','DelaiInt_2', 'PanneDelai_2']
discrete_lst = [False,True,
                True,True,True,True,False,
                True, True, True, True, False,False]
mi_MI = mutual_info_regression(ArmInt_encode[score_lst], ArmInt_encode['PanneDelai_1'],discrete_features=discrete_lst)
df_mi_MI = pd.DataFrame(columns=['feature','score'])
df_mi_MI['feature'] = score_lst
df_mi_MI['score'] = mi_MI
saver.save_excel(data=df_mi_MI,filename='feature_scoreMI_Armoire',foldername='Encode')

mi_F, pvalue_F = f_regression(ArmInt_encode[score_lst], ArmInt_encode['PanneDelai_1'],center=True)
print(mi_F)
df_mi_F = pd.DataFrame(columns=['feature','score','pvalue'])
df_mi_F['feature'] = score_lst
df_mi_F['score'] = mi_F
df_mi_F['pvalue'] = pvalue_F
saver.save_excel(data=df_mi_F,filename='feature_scoreF_Armoire',foldername='Encode')





### PL:
## ADD NEW VARS
data_merge_select_PL = loader.load_excel(filename='merge_PLInt')
new_merge_PLInt = processor.create_newvar(data_merge=data_merge_select_PL)
saver.save_excel(data=new_merge_PLInt,filename='merge_PLInt_addnewvar',foldername='Encode')
new_PLInt_drop = new_merge_PLInt.dropna(subset=['PanneDelai_1'])
saver.save_excel(data=new_PLInt_drop,filename='merge_PLInt_dropna_PanneDelai_1',foldername='Encode')

## encode




"""
Rank the vars according to IG
"""



"""
Clustering and visualization of the characteristics of the clusters
"""