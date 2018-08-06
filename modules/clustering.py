import os
# import pandas as pd
# from docx import Document
import pandas as pd
# import matplotlib.pyplot as plt
# from docx.shared import Inches
# import seaborn as sns
# from utils.constants import Armoire_GROUP,Armoire_PICK
# from utils.constants import PL_GROUP,PL_PICK
# from utils.constants import Int_GROUP,Int_PICK
from utils.constants import VILLE_NAME
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


class Cluster(object):
    def __init__(self, datestr,datasavedir=""):
        if datasavedir == "":
            self.basedir = os.path.abspath(os.path.dirname(__file__))
            self.datasavedir = os.path.abspath(
                os.path.join(self.basedir, '..','..','data_save{}'.format(datestr)))
        else:
            self.datasavedir = datasavedir

    def add_Int(self, data_row, data_Int, int_num=3,Var_Cat= {'pan_Solde':3, 'int_Solde':4}, Var_Num=['pan_DateSignal','int_DateIntervention','int_Fin']):
        """

        :param data_row:
        :param data_Int:
        :param int_num:
        :param one_hot:
        :param Var_Cat: a dict which denotes the categorical variables and the number of categories
        :param Var_Num: a list of numerical variables which have preprocessed (add NA, add clip)
        :return:
        """
        assert isinstance(Var_Cat, dict), 'Var_Cat should be a dict !'
        assert isinstance(Var_Num,list), 'Var_Num should be a list !'

        new_data = data_row.copy()
        data_pick = data_Int.loc[(data_Int['pan_CodeEqt'] == new_data['eq_Code']) & (data_Int['region'] == new_data['region'])]

        # attention! if concanate eqt with intervention, deal with the index!
        data_int = self.reorder_Int(data=data_pick, int_num=int_num, Var_Cat=Var_Cat, Var_Num=Var_Num)
        data_int_series = data_int.iloc[0]

        # add the info of intervention (turned to string)
        data_pick_str = pd.Series([data_pick[list(Var_Cat.keys())+Var_Num].to_string()], index=['int_Info'])


        new_data = pd.concat([data_pick_str, new_data, data_int_series], axis=0)

        return new_data


    def getcontent_kmeans_index(self, data, k_means, cluster_index,colnames):
        """
        Get the emails from the cluster index (after Kmeans)

        Parameters
        ----------
        data: pd.DataFrame
            DataFrame which contains the content
        k_means: sklearn.cluster.KMeans
            the KMeans object that we get from the function "implement_kmeans()"
        cluster_index:int
            the index of the clusters_kmeans
        Returns
        -------
        cluster_size: int
            the number of mails in this cluster
        count_tf:sklearn.feature_extraction.text.CountVectorizer.fit_transform()

        result: pd.DataFrame
            the new dataframe that we create for storageing the mails of this cluster
            (columns:'cluster_index','cluster_size','id','content')
        """

        # the index of the mails which belong to the cluster
        index_data = np.where(k_means.labels_ == cluster_index)[0].tolist()
        # print('merge index==========')
        # print(type(index_data))
        # print(index_data)

        cluster_size = len(index_data)
        # newdf = pd.concat([data['id'][index_data], data['content'][index_data]], axis=1)
        newdf = data.ix[index_data,colnames]

        newdf.reset_index(drop=True, inplace=True)
        index_size_df = pd.DataFrame({'cluster_index': [cluster_index for i in range(cluster_size)],
                                      'cluster_size': [cluster_size for i in range(cluster_size)]})
        index_size_df.reset_index(drop=True, inplace=True)
        result = pd.concat([index_size_df, newdf], axis=1)
        return cluster_size, result

    def getcontent_kmeans(self, data, k_means,colnames):
        """
        group the the emails of all the clusters(after Kmeans) using the function
        "getmail_kmeans_index()"

        Parameters
        ----------
        data: pd.DataFrame
            DataFrame which contains the content
        k_means: sklearn.cluster.KMeans
            the KMeans object that we get from the function "implement_kmeans()"

        Returns
        -------
        clustersize_list: list
            a list which contains the number of mails in each cluster
        result: pd.DataFrame
            the new dataframe that we create for storageing the mails of this cluster
            (columns:'cluster_index','cluster_size','id','content')
        """
        n_cluster = len(k_means.cluster_centers_)
        df = pd.DataFrame(columns=['cluster_index', 'cluster_size']+colnames)
        clustersize_lst = []
        for i in range(n_cluster):
            cluster_size, dfcluster = self.getcontent_kmeans_index(data, k_means, i,colnames)
            clustersize_lst.append(cluster_size)
            df = df.append(dfcluster, ignore_index=True)
        return clustersize_lst, df

    def getcontent_HC_index(self, data, link_matrix, k,cluster_index,colnames):
        # the index of the mails which belong to the cluster
        HClabels = fcluster(link_matrix, k, criterion='maxclust')
        index_data = np.where((HClabels-1) == cluster_index)[0].tolist()

        cluster_size = len(index_data)
        newdf = data.ix[index_data,colnames]

        newdf.reset_index(drop=True, inplace=True)
        index_size_df = pd.DataFrame({'cluster_index': [cluster_index for i in range(cluster_size)],
                                      'cluster_size': [cluster_size for i in range(cluster_size)]})
        index_size_df.reset_index(drop=True, inplace=True)
        result = pd.concat([index_size_df, newdf], axis=1)
        return cluster_size, result

    def getcontent_HC(self, data, link_matrix, k, colnames):
        n_cluster = k
        df = pd.DataFrame(columns=['cluster_index', 'cluster_size']+colnames)
        clustersize_lst = []
        for i in range(n_cluster):
            cluster_size, dfcluster = self.getcontent_HC_index(data, link_matrix, n_cluster, i,colnames)
            clustersize_lst.append(cluster_size)
            df = df.append(dfcluster, ignore_index=True)
        return clustersize_lst, df

    def aggregateHC(self, link_matrix, k, feature_matrix):
        """
        Aggregation of the result of Hierarchical Clustering

        Parameters
        ----------
        link_matrix:
            the result of "linkage(feature_matrix)"
        depth: int
            the number of clusters that we want
        feature_matrix:
            feature matrix
        Returns
        -------
        labelindex_H: list
            list of indices which belong to the same cluster
        newfeature_H: matrix
            the new feature matrix after Hierarchical Clustering
        """

        # ----------------------HC--------------
        clusterlabels = fcluster(link_matrix, k, criterion='maxclust')
        labelindex_H = []  # list of indices which belong to the same cluster
        labelindex_H.append(list(np.where(clusterlabels == 1)))
        newfeature_H = np.mean(feature_matrix[labelindex_H[0]],
                               axis=0)  # aggregation according to the label result of Hierarchical Clustering
        for i in range(2, k + 1):
            labelindex_H.append(list(np.where(clusterlabels == i)))
            newfeature_H = np.row_stack((newfeature_H, np.mean(feature_matrix[labelindex_H[i - 1]], axis=0)))
        # ---------------------------
        # cluster_labels = k_means.labels_
        # for i in range(0, n_kmeans):
        #     cluster_labels[np.where(k_means.labels_ == i)] = hcluster.labels_[i]
        return labelindex_H, newfeature_H

    def getcontent_mixed(self, df_kmeans, link_matrix, k, colnames):
        """
        group the the emails of all the clusters(after mixed classification) using the function
        "getmail_mixed_index()"

        Parameters
        ----------
        df_kmeans: pd.DataFrame
            the result of regrouping after Kmeans (the result of the function "getmail_kmeans()")
        link_matrix: scipy.cluster.hierarchy.linkage object
            it is for retrieving the labels
        k: int
            the number of clusters of Hierarchical clustering
        Returns
        -------
        clustersize_lst: list
            a list which contains the number of mails in each cluster

        result: pd.DataFrame
            the new dataframe that we create for storageing the mails of this cluster
            (columns:'cluster_index','cluster_size','id','content')
        """
        n_cluster = k
        df = pd.DataFrame(columns=['cluster_index', 'cluster_size']+colnames)
        clustersize_lst = []
        for i in range(n_cluster):
            cluster_size, dfcluster = self.getcontent_mixed_index(df_kmeans, link_matrix, k, i,colnames)
            clustersize_lst.append(cluster_size)
            df = df.append(dfcluster, ignore_index=True)
            # df=pd.concat([df,dfcluster], ignore_index=True)
        return clustersize_lst, df

    def getcontent_mixed_index(self, df_kmeans, link_matrix, k, index_HC,colnames):
        """
        Get the emails from the cluster index (after mixed classification)

        Parameters
        ----------
        df_kmeans: pd.DataFrame
            the result of regrouping after Kmeans (the result of the function "getmail_kmeans()")
        index_HC:int
            the index of the clusters_Hierarchical clustering
        link_matrix: scipy.cluster.hierarchy.linkage object
            it is for retrieving the labels
        k: int
            the number of clusters of Hierarchical clustering
        Returns
        -------
        cluster_size: int
            the number of elements in this cluster

        result: pd.DataFrame
            the new dataframe that we create for storageing the mails of this cluster
            (columns:'cluster_index','cluster_size','id','content')
        """
        HClabels = fcluster(link_matrix, k, criterion='maxclust')
        index_data = np.where(HClabels-1 == index_HC)[0].tolist()
        arr_kmeans = df_kmeans['cluster_index'].values
        index_contentlst = []
        cluster_size = 0
        for i in index_data:
            lstmp = np.where(arr_kmeans == i)[0].tolist()
            cluster_size += len(lstmp)
            index_contentlst.extend(lstmp)

        # newdf = pd.concat([df_kmeans['id'][index_contentlst], df_kmeans['content'][index_contentlst]], axis=1)
        newdf = df_kmeans.ix[index_contentlst,colnames]

        newdf.reset_index(drop=True, inplace=True)
        index_size_df = pd.DataFrame({'cluster_index': [index_HC for i in range(cluster_size)],
                                      'cluster_size': [cluster_size for i in range(cluster_size)]})
        index_size_df.reset_index(drop=True, inplace=True)
        return cluster_size, pd.concat([index_size_df, newdf], axis=1)
