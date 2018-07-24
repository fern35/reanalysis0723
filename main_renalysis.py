from modules.loader import Loader
from modules.saver import Saver
from modules.analyser import Analyzer
from modules.cleaner import Cleaner
from modules.preprocessor import Processor
from modules.preprocessor import Processor
import datetime as dt
from utils.constants import VILLE_NAME,Armoire_NAME,PL_NAME,Int_NAME,Armoire_TIME,PL_TIME,Int_TIME


# the date of saving the data
date_str='0723'
analyzer = Analyzer(datestr=date_str)
cleaner = Cleaner()
loader = Loader()
saver = Saver(datestr=date_str)
processor = Processor(datestr=date_str)

data_downloadtime = dt.datetime(2018, 5, 15, 0, 0, 0, 0)
data_starttime = dt.datetime(2013, 1, 1, 0, 0, 0, 0)
day_difference = (data_downloadtime - data_starttime).days

CURRENT_TIME_AP = '2018-05-15'
CURRENT_TIME_INT = '2018_05_15'
Intfilename_lst = ["BDDExportInterventions-{} du 01_01_2013 au 15_05_2018.xlsx".format(CURRENT_TIME_INT)]


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
#
#
# data_Armoire = processor.merge_file(foldername='Armoire', villelst=VILLE_NAME, add_region=True)
# saver.save_excel(data_Armoire, foldername='Armoire',filename='Armoire_allcities')
#
# data_PL = processor.merge_file(foldername='PL', villelst=VILLE_NAME, add_region=True)
# saver.save_excel(data_PL, foldername='PL', filename='PL_allcities')
#
# data_Int = processor.merge_file(foldername='Int', villelst=VILLE_NAME, add_region=True)
# saver.save_excel(data_Int, foldername='Int', filename='Int_allcities')


"""
Merge Intervantion with Armoire and PL, 
before doing that, check the distribution of the count of interventions
==> for identifying the number of interventions for loading to the merged file
"""
# check 





"""
encode categorical vars and standardize numerical vars, 
attention: need to deal with the vars with many categories and NLP vars
"""



"""
Rank the vars according to IG
"""



"""
Clustering and visualization of the characteristics of the clusters
"""