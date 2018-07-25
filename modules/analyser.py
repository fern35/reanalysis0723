import os
from docx import Document
import pandas as pd
import matplotlib.pyplot as plt
from docx.shared import Inches
import seaborn as sns
from utils.constants import Armoire_GROUP,Armoire_PICK
from utils.constants import PL_GROUP,PL_PICK
from utils.constants import Int_GROUP,Int_PICK
from utils.constants import VILLE_NAME
from modules.saver import Saver
from sklearn.cluster import KMeans
import numpy as np

class Analyzer(object):
    def __init__(self, datestr, datasavedir=""):
        if datasavedir == "":
            self.basedir = os.path.abspath(os.path.dirname(__file__))
            self.datasavedir = os.path.abspath(
                os.path.join(self.basedir, '..','..','data_save{}'.format(datestr)))
        else:
            self.datasavedir = datasavedir
        self.saver = Saver(datestr=datestr)

    def pick_Var(self,data, Var_lst):
        new_data = data[Var_lst]
        return new_data

    def check_savepath(self,foldpath, filename):
        if not os.path.exists(foldpath):
            os.makedirs(foldpath)
        save_path = os.path.join(foldpath,filename)
        try:
            os.remove(save_path)
        except OSError:
            pass
        return save_path

    def gen_NAN_excel(self,data,foldername='Armoire_arm',titlename='Armoire_arm',save=True):
        """generate data frame describing the condition of NAN for variables in one file"""
        stat_data = pd.DataFrame(columns=['count_wnNAN'])
        stat_data['count_wnNAN'] = data.count()
        stat_data['percentage_wnNAN'] = data.count() / len(data)
        if save:
            self.saver.save_excel(data=stat_data,filename='{}_NAN.xlsx'.format(titlename),foldername=foldername)
            # fold_path = os.path.join(self.datasavedir,'excel','{}'.format(foldername))
            # save_path = self.check_savepath(foldpath=fold_path,filename='{}_NAN.xlsx'.format(titlename))
            # writer = pd.ExcelWriter(save_path)
            # stat_data.to_excel(writer, 'Sheet1')
            # writer.save()
        return stat_data

    def gen_cat_var(self, data, Var_lst):
        """
        check all the categories of the variables of the data
        :param data:
        :param Var_lst:
        :return:
        """
        result_data = pd.DataFrame(columns=Var_lst)
        for var in Var_lst:
            tp = data[var].value_counts(dropna=False)
            result_data[var] = pd.Series(tp.index)

        return result_data

    def get_tfvect_feature(self,vectorizer,save=True,foldername='',varname=''):
        df_feature = pd.DataFrame(columns=['feature','idf'])
        df_feature['feature'] = vectorizer.get_feature_names()
        def get_idf(feature):
            index_feature = vectorizer.vocabulary_[feature]
            return vectorizer.idf_[index_feature]
        df_feature['idf'] = df_feature['feature'].apply(get_idf)
        df_feature.sort_values(by='idf',inplace=True)
        if save:
            self.saver.save_excel(data=df_feature,filename='Idf_Analysis_{}.docx'.format(varname),foldername=foldername)
        return df_feature

    def ana_gapstat_kmeans(self,n_cluster_max,data_matrix,titlename,nrefs=10):

        gaps = np.zeros((len(range(1, n_cluster_max)),))
        resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
        for gap_index, k in enumerate(range(1, n_cluster_max)):
            # Holder for reference dispersion results
            refDisps = np.zeros(nrefs)
            # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
            for i in range(nrefs):
                # Create new random reference set
                randomReference = np.random.random_sample(size=data_matrix.shape)
                # Fit to it
                km = KMeans(k)
                km.fit(randomReference)
                refDisp = km.inertia_
                refDisps[i] = refDisp
            # Fit cluster to original data and create dispersion
            km = KMeans(k)
            km.fit(data_matrix)
            origDisp = km.inertia_
            # Calculate gap statistic
            gap = np.log(np.mean(refDisps)) - np.log(origDisp)
            # Assign this loop's gap statistic to gaps
            gaps[gap_index] = gap
            resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, ignore_index=True)

        optimalK = gaps.argmax() + 1

        plt.plot(resultsdf.clusterCount, resultsdf.gap, linewidth=3)
        plt.scatter(resultsdf[resultsdf.clusterCount == optimalK].clusterCount, resultsdf[resultsdf.clusterCount == optimalK].gap, s=250, c='r')
        plt.grid(True)
        plt.xlabel('Cluster Count')
        plt.ylabel('Gap Value')
        plt.title(titlename)
        fold_path = os.path.join(self.datasavedir, 'img', 'inertia')
        save_path = self.check_savepath(foldpath=fold_path,filename='{}.jpg'.format(titlename))
        plt.savefig(save_path)
        plt.clf()

        return optimalK, resultsdf

    def gen_groupCompl_cities(self,foldername='Armoire', villelst=VILLE_NAME,group_dict=Armoire_GROUP,threshold=0.0):
        document = Document()
        document.add_heading('{}_completeness_threshold={}'.format(foldername,threshold), 0)

        for ville in villelst:
            data = pd.read_excel(os.path.join(self.datasavedir, 'excel/{}/{}_{}.xlsx'.format(foldername,foldername,ville)))
            data_groupNAN = self.group_Compl(data, group_dict=group_dict,threshold=threshold)
            plt.clf()
            data_groupNAN.plot(kind='bar',figsize=(3,3),fontsize=8,legend=False)
            plt.title('completeness variables {}_threshold={}'.format(ville,threshold), fontsize=8)
            try:
                plt.tight_layout()
            except:
                print('Cannot use tight layout, var: ',ville)

            plt.ylabel('percentage vars >{}'.format(threshold))
            plt.savefig(os.path.abspath(os.path.join(self.datasavedir, 'img/groupCompl/{}_{}_groupCompl_threshold{}.jpg'.format(foldername,ville,threshold))))
            document.add_picture(os.path.abspath(os.path.join(self.datasavedir, 'img/groupCompl/{}_{}_groupCompl_threshold{}.jpg'.format(foldername,ville,threshold))))

        fold_path = os.path.join(self.datasavedir, 'doc', '{}'.format(foldername))
        save_path = self.check_savepath(foldpath=fold_path, filename='groupCompl{}.docx'.format(threshold))
        document.save(save_path)

    def group_Compl(self,data,group_dict,threshold):
        """generate completeness condition of each group for one city"""
        result = pd.DataFrame(index=group_dict.keys())
        freq = data.count() / len(data)
        for group,vars in group_dict.items():
            count_group = 0
            count_threshold = 0
            for var in vars:
                if var in freq.index.tolist() :
                    count_group += 1
                    if freq.loc[var]>threshold:
                        count_threshold += 1
            result.loc[group,0] = count_threshold/count_group
        return result

    def gen_VarIntersection(self,foldername='Armoire',villelst=VILLE_NAME,group_dict=Armoire_GROUP,threshold=0.0):
        result_dict = dict.fromkeys(group_dict.keys(),[])
        for ville in villelst:
            data = pd.read_excel(os.path.abspath(os.path.join(self.datasavedir, 'excel/{}/{}_{}.xlsx'.format(foldername,foldername,ville))))
            ville_dict = self.gen_VarThreshold(data,group_dict=group_dict,threshold=threshold)

            for group in group_dict:
                lst = result_dict[group].copy()
                lst.append(ville_dict[group])
                result_dict[group] = lst

        for group in group_dict:
            result_dict[group] = set.intersection(*map(set, result_dict[group]))

        # save to txt
        f = open(self.datasavedir+'/doc/{}/var_threshold{}.txt'.format(foldername,threshold), 'w')
        for key, value in result_dict.items():
            f.write(key + ':' + str(value))
            f.write('\n')
        f.close()
        return result_dict

    def gen_VarThreshold(self,data,group_dict,threshold=0.0):
        result_dict = dict.fromkeys(group_dict.keys(),[])
        stat_data = self.gen_NAN_excel(data,save=False)
        for group,vars in group_dict.items():
            lst_tp = []
            for var in vars:
                if stat_data.loc[var,'percentage_wnNAN']>threshold:
                    lst_tp.append(var)
            result_dict[group] = lst_tp
        return result_dict

    def comp_Var_cities(self,foldername='Armoire',villelst=VILLE_NAME,group_dict= Armoire_PICK):
        # first test the result of categorical variables
        document = Document()
        document.add_heading('compa_Var_{}'.format(foldername), 0)
        var_lst = group_dict['CAT']+group_dict['DIST']
        for var in var_lst:
            if var in group_dict['CAT']:
                result_cat = pd.DataFrame(columns=villelst)
                for ville in villelst:
                    data = pd.read_excel(os.path.join(self.datasavedir, 'excel','{}'.format(foldername),'{}_{}.xlsx'.format(foldername,ville)))
                    result_cat[ville] = data[var].value_counts(dropna=False)/len(data)

                # if high variety of values, split the graph
                if len(result_cat)>8:
                    no_split = int(len(result_cat)/8)
                    for i in range(no_split):
                        plt.clf()
                        result_cat.iloc[i*8:i*8+7,:].plot(kind='bar')
                        plt.title('{} for cities_BDD{}_{}:{}'.format(var, foldername,i*8,i*8+7), fontsize=12)
                        try:
                            plt.tight_layout()
                        except:
                            print('Cannot use tight layout, var: ', var)

                        plt.ylabel('percentage')
                        fold_path_mul = os.path.join(self.datasavedir,'img','varDistCities')
                        save_path_mul = self.check_savepath(foldpath=fold_path_mul,filename='{}_{}_{}:{}.jpg'.format(foldername, var,i*8,i*8+7))
                        plt.savefig(save_path_mul)
                        document.add_picture(save_path_mul,
                                             width=Inches(4.5))
                    plt.clf()
                    result_cat.iloc[no_split*8:,:].plot(kind='bar')
                    plt.title('{} for cities_BDD{}_{}:{}'.format(var, foldername,no_split*8,len(result_cat)-1), fontsize=12)
                    try:
                        plt.tight_layout()
                    except:
                        print('Cannot use tight layout, var: ', var)

                    plt.ylabel('percentage')

                    fold_path_last = os.path.join(self.datasavedir, 'img', 'varDistCities')
                    save_path_last = self.check_savepath(foldpath=fold_path_last,
                                                        filename='{}_{}_{}:{}.jpg'.format(foldername, var,no_split*8,len(result_cat)-1))
                    plt.savefig(save_path_last)
                    document.add_picture(save_path_last,width=Inches(4.5))

                else:
                    plt.clf()
                    result_cat.plot(kind='bar')
                    plt.title('{} for cities_BDD{}'.format(var,foldername), fontsize=12)
                    try:
                        plt.tight_layout()
                    except:
                        print('Cannot use tight layout, var: ',var)

                    plt.ylabel('percentage')
                    fold_path_img = os.path.join(self.datasavedir, 'img', 'varDistCities')
                    save_path_img = self.check_savepath(foldpath=fold_path_img,
                                                        filename='{}_{}.jpg'.format(foldername,var))

                    plt.savefig(save_path_img)
                    document.add_picture(save_path_img,
                                         width=Inches(4.5))
            else:
                result_dist = pd.DataFrame(columns=villelst)
                for ville in villelst:
                    data = pd.read_excel(os.path.join(self.datasavedir, 'excel','{}'.format(foldername),'{}_{}.xlsx'.format(foldername,ville)))
                    result_dist[ville] = data[var]

                plt.clf()
                result_dist.plot(kind='density')
                plt.title('{} for cities_BDD{}'.format(var, foldername), fontsize=12)
                try:
                    plt.tight_layout()
                except:
                    print('Cannot use tight layout, var: ', var)

                plt.ylabel('percentage')
                fold_path_img_var = os.path.join(self.datasavedir, 'img', 'varDistCities')
                save_path_img_var = self.check_savepath(foldpath=fold_path_img_var,filename='{}_{}.jpg'.format(foldername, var))
                plt.savefig(save_path_img_var)
                document.add_picture(save_path_img_var,
                                     width=Inches(4.5))

        fold_path_doc = os.path.join(self.datasavedir,'doc','{}'.format(foldername))
        save_path_doc = self.check_savepath(foldpath=fold_path_doc,filename='compa_Var_{}.docx'.format(foldername))
        document.save(save_path_doc)

    def gen_histogram_Pie(self,data,titlename,Var_lst):
        document = Document()

        document.add_heading('Histogram & Pie_{}_{} instances'.format(titlename,len(data)), 0)

        for var in Var_lst:
            plt.clf()
            data[var].value_counts(dropna=False).plot(kind='bar')
            plt.title('Histogram {}'.format(var), fontsize=12)
            try:
                plt.tight_layout()
            except:
                print('Cannot use tight layout, var: ',var)

            plt.ylabel('numbers')
            fold_path_bar = os.path.join(self.datasavedir,'img','{}'.format(titlename))
            save_path_bar = self.check_savepath(foldpath=fold_path_bar,filename='{}_{}_bar.jpg'.format(titlename,var))
            plt.savefig(save_path_bar)
            document.add_picture(save_path_bar,width=Inches(4.5))

            plt.clf()
            (data[var].value_counts(dropna=False)/len(data)).plot(kind='pie')
            plt.title('Pie plot {}'.format(var), fontsize=12)
            plt.tight_layout()
            fold_path_pie = os.path.join(self.datasavedir,'img','{}'.format(titlename))
            save_path_pie = self.check_savepath(foldpath=fold_path_pie,filename='{}_{}_pie.jpg'.format(titlename,var))
            plt.savefig(save_path_pie)
            document.add_picture(save_path_pie,width=Inches(4.5))

        fold_path_doc = os.path.join(self.datasavedir,'doc','{}'.format(titlename))
        save_path_doc = self.check_savepath(foldpath=fold_path_doc,filename='Histogram_Pie_{}.docx'.format(titlename))
        document.save(save_path_doc)

    def gen_Dist(self,data,titlename,Var_lst):
        document = Document()

        document.add_heading('Distribution_{}_{} instances'.format(titlename,len(data)), 0)

        for var in Var_lst:
            plt.clf()
            sns.distplot(data[var].dropna())
            plt.title('Distribution(dropna={}) {}'.format(data[var].isnull().sum(),var), fontsize=12)
            plt.ylabel('frequncy')
            plt.tight_layout()
            fold_path_img = os.path.join(self.datasavedir,'img','{}'.format(titlename))
            save_path_img = self.check_savepath(foldpath=fold_path_img,filename='{}_{}_dist.jpg'.format(titlename,var))
            plt.savefig(save_path_img)
            document.add_picture(save_path_img,width=Inches(4.5))

        fold_path_doc = os.path.join(self.datasavedir,'doc','{}'.format(titlename))
        save_path_doc = self.check_savepath(foldpath=fold_path_doc,filename='Dist_{}.docx'.format(titlename))
        document.save(save_path_doc)
