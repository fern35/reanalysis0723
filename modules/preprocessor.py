import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os
from utils.constants import VILLE_NAME
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
class Processor(object):
    def __init__(self, datestr,datasavedir=""):
        if datasavedir == "":
            self.basedir = os.path.abspath(os.path.dirname(__file__))
            self.datasavedir = os.path.abspath(
                os.path.join(self.basedir, '..','..','data_save{}'.format(datestr)))
        else:
            self.datasavedir = datasavedir

    def merge_file(self, foldername='Armoire', villelst=VILLE_NAME, add_region=True, Var_lst=None):
        # retrieve the column names
        fold_path = os.path.join(self.datasavedir, 'excel','{}'.format(foldername))
        data_forcol = pd.read_excel(
            os.path.join(fold_path,'{}_{}.xlsx'.format(foldername, villelst[0])))
        # get the names of columns
        merge_data = pd.DataFrame(columns=data_forcol.columns)

        for ville in villelst:
            data_tp = pd.read_excel(
                os.path.join(fold_path,'{}_{}.xlsx'.format( foldername, ville)))
            # because the column 'commune' in the database are not complete
            data_tp['region'] = ville
            merge_data = pd.concat([merge_data, data_tp])

        if Var_lst is not None:
            merge_data = merge_data[Var_lst+['region']]
        if not add_region:
            merge_data.drop(['region'], axis=1, inplace=True)

        merge_data.drop_duplicates(inplace=True)
        merge_data = merge_data.reset_index( drop=True)
        return merge_data

    def var_vectorize(self,data,vectorizer=TfidfVectorizer(strip_accents='unicode',stop_words=stopwords.words('french'))):
        # fillna
        new_data = data.fillna('NA')
        # remove figures? or replace figures by 0? or keep figures?

        df_vect = vectorizer.fit_transform(new_data)
        return df_vect,vectorizer

    def NL_encode(self,data, var, vectorizer=TfidfVectorizer(strip_accents='unicode',stop_words=stopwords.words('french'))):
        """
        encode NL variables with the personalized vectorizer
        :param data:
        :param var:
        :return:
        """
        newdata = data.copy()
        newdata[var + '_encode'] = self.var_vectorize(data=newdata[var],vectorizer=vectorizer)
        return newdata


    def cat_encode(self, data, var, regroup_dict=None):
        """
        encode categorical variables with the personalized dictionary
        :param data:
        :param var:
        :param one_hot:
        :param regroup_dict: dict(new class: [old classes])
        :return:
        """
        new_data = data.fillna(value={var:'NA'})
        grouped = new_data.groupby(new_data[var])
        classes = list(grouped.groups.keys())
        def rep_vsdict(key,dict):
            return dict[key]

        if regroup_dict is None:
            dict_encode = dict(zip(classes, range(len(classes))))
            new_data[var+'_encode'] = new_data[var].apply(lambda x: rep_vsdict(x, dict=dict_encode))
        else:
            dict_encode = dict(zip(regroup_dict.keys(), range(len(regroup_dict))))
            dict_encode_ = dict.fromkeys(classes, None)
            for key, values in regroup_dict.items():
                for value in values:
                    assert value in classes, "The value: {} of group_dict does not correspond to the actual classes".format(value)
                    dict_encode_[value] = dict_encode[key]
            new_data[var+'_encode'] = new_data[var].apply(lambda x: rep_vsdict(x, dict=dict_encode_))

        return dict_encode, new_data

    def num_encode(self, data, var, proper_range=None):
        new_data = data.copy()
        new_data[var+'_NA'] = new_data[var].isnull().apply(lambda x: x*1)
        new_data = new_data.fillna(value={var: 0})
        if proper_range is not None:
            new_data[var+'_clip'] = new_data[var].clip(proper_range[0], proper_range[1])
        return new_data

    def turn_onehot(self,data, var, n_len):
        """
        applied for one eqt
        :param data:
        :param var:
        :param n_len:
        :return:
        """
        # assert data.empty, 'the data for turning to one hot cannot be empty!'
        new_data = data.copy()
        onehot_encoder = OneHotEncoder()

        onehot_encoder.fit(X=np.asarray(range(n_len))[:,np.newaxis])
        array_encoded = onehot_encoder.transform(X=new_data[var].values.reshape(-1, 1)).toarray()

        # col_num = len(array_encoded[0])
        col_name = [var+'_code'+str(i) for i in range(n_len)]
        df_encoded = pd.DataFrame(array_encoded,columns=col_name)
        new_data.reset_index(drop=True, inplace=True)
        new_data = pd.concat([new_data, df_encoded],axis=1)

        return new_data

    def reorder_Int(self, data, int_num, Var_Cat, Var_Num):
        Var_Cat_encode = Var_Cat.copy()
        for var in Var_Cat:
            Var_Cat_encode[var+'_encode'] = Var_Cat_encode.pop(var)

        cat_lst = list(Var_Cat_encode.keys())
        num_lst = [var+'_NA' for var in Var_Num]+[var+'_clip' for var in Var_Num]
        col_lst_or = cat_lst+num_lst
        assert set(data.columns) > set(col_lst_or), 'Some variables in par:Var_Cat & par:Var_Num are not in the data !'
        col_lst_encode = []
        # create empty df
        for var, num_cat in Var_Cat_encode.items():
            lst_tp = [var + '_code' + str(i) for i in range(num_cat)]
            col_lst_encode += lst_tp
        col_lst_encode += num_lst
        col_lst_final = [var+'_last'+str(i)
            for i in range(int_num)
            for var in col_lst_encode
            ]
        new_data = pd.DataFrame(0, index=[0], columns=col_lst_final)

        if data.empty:
            return new_data
        else:
            if len(data)>int_num:
                data_copy = data.copy().reset_index(drop=True)
                data_copy = data_copy.ix[0:int_num-1,col_lst_or]
                assert len(data_copy)<=int_num, 'The length of data_copy is greater than {}'.format(int_num)
                assert not data_copy.empty, 'The data_copy cannot be empty! col_lst_or=={}'.format(col_lst_or)
            else:
                data_copy = data.copy().reset_index(drop=True)
                data_copy = data_copy.loc[:,col_lst_or]
                assert not data_copy.empty, 'The data_copy cannot be empty! len(data)<=int_num'

            for var,num_cat in Var_Cat_encode.items():
                data_copy = self.turn_onehot(data=data_copy,var=var,n_len=num_cat)

            data_copy = data_copy.sort_values(by=num_lst)
            # print('data_copy.columns=============')
            # print(data_copy.columns)
            # here, data_copy have several rows, with original cols like

            data_tp = data_copy[col_lst_encode]

            array_insert = np.squeeze(data_tp.values.reshape(1,-1))
        # insert values to empty df
        #     print('len{}_{}'.format(len(array_insert),array_insert))
            new_data.ix[0, 0:len(array_insert)] = array_insert
            return new_data