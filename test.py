from modules.analyser import Analyzer
from modules.loader import Loader
import numpy as np

date_str = '0723'
analyzer = Analyzer(datestr=date_str)
loader = Loader(date_str)


# ArmInt_cluster = loader.load_excel(filename='ArmInt_cluster',foldername='Cluster')
# ArmInt_cluster.drop(['PanneDelai_1'], axis=1,inplace=True)
# feature_names = np.array(list(ArmInt_cluster.columns))
#
# clf = loader.load_pickle('Randomforest_Armoire')
# analyzer.plot_feature_importance(importances=clf.best_estimator_.feature_importances_,featurenames=feature_names,title='Randomforest_featureimportance_Armoire',top_n=40)
#
# clf = loader.load_pickle('GradientBoosting_Armoire')
# analyzer.plot_feature_importance(importances=clf.best_estimator_.feature_importances_,featurenames=feature_names,title='GradientBoosting_featureimportance_Armoire',top_n=40)
#



PL_cluster = loader.load_excel(filename='PL_cluster',foldername='Cluster')
PL_cluster.drop(['PanneDelai_1'], axis=1,inplace=True)
feature_names = np.array(list(PL_cluster.columns))

clf = loader.load_pickle('Randomforest_PL')
analyzer.plot_feature_importance(importances=clf.best_estimator_.feature_importances_,featurenames=feature_names,title='Randomforest_featureimportance_PL',top_n=40)

clf = loader.load_pickle('GradientBoosting_PL')
analyzer.plot_feature_importance(importances=clf.best_estimator_.feature_importances_,featurenames=feature_names,title='GradientBoosting_featureimportance_PL',top_n=40)




