import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import string
import re
import os
from utils.constants import VILLE_NAME
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from utils.constants import Int_MERGE
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

        tfidf_vect = vectorizer.fit_transform(new_data)
        return tfidf_vect,vectorizer

    # def NL_encode(self,data, var, vectorizer=TfidfVectorizer(strip_accents='unicode',stop_words=stopwords.words('french'))):
    #     """
    #     encode NL variables with the personalized vectorizer
    #     :param data:
    #     :param var:
    #     :return:
    #     """
    #     newdata = data.copy()
    #     newdata[var + '_encode'] = self.var_vectorize(data=newdata[var],vectorizer=vectorizer)
    #     return newdata

    def rep_vsdict(self,key,dict):
        """
        used in cat_encode
        :param dict:
        :return:
        """
        return dict[key]

    def cat_encode(self, data, var, regroup_dict=None):
        """
        encode categorical variables with the personalized dictionary,
        attention: we replace the values of the variable
        :param data:
        :param var:
        :param one_hot:
        :param regroup_dict: dict(new class: [old classes])
        :return:
        """
        new_data = data.fillna(value={var:'NA'})
        grouped = new_data.groupby(new_data[var])
        classes = list(grouped.groups.keys())

        if regroup_dict is None:
            dict_encode = dict(zip(classes, range(len(classes))))
            # new_data[var+'_encode'] = new_data[var].apply(lambda x: rep_vsdict(x, dict=dict_encode))
            new_data[var] = new_data[var].apply(lambda x: self.rep_vsdict(x, dict=dict_encode))
        else:
            dict_encode = dict(zip(regroup_dict.keys(), range(len(regroup_dict))))
            dict_encode_ = dict.fromkeys(classes, None)
            for key, values in regroup_dict.items():
                for value in values:
                    #assert value in classes, "The value: {} of group_dict does not correspond to the actual classes".format(value)
                    dict_encode_[value] = dict_encode[key]
            # new_data[var+'_encode'] = new_data[var].apply(lambda x: rep_vsdict(x, dict=dict_encode_))
            new_data[var] = new_data[var].apply(lambda x: self.rep_vsdict(x, dict=dict_encode_))

        return dict_encode, new_data

    def cat_multi_encode_onerow(self,data,var,regroup_dict):
        """
        the result is one hot
        :param data:
        :param pattern:
        :param encode_dict: example_dct = {'CAT1':0, 'CAT2':0, 'CAT3':1, ...}
        :return:
        """
        def find_newcat(category,dict):
            for key, value_lst in dict.items():
                if category in value_lst:
                    return key
                else:
                    # print('cannot find the \'{}\' in the value list \'{}\'!'.format(category,value_lst))
                    pass

        cat_lst = data[var].split('\n')
        # cat_lst = [self.check_content_NL(i, pattern) for i in cat_lst]
        colnames = [var+'_'+ele for ele in regroup_dict.keys()]
        result_dict = dict.fromkeys(colnames,0)

        for cat in cat_lst:
            # find the key in regroup_dict
            # print('category: ',cat)
            # print(regroup_dict)
            # print('find: ', find_newcat(category=cat,dict=regroup_dict))
            # print("=====================")
            result_dict[var+'_'+find_newcat(category=cat,dict=regroup_dict)] = 1
        result_data = pd.Series(result_dict)

        return result_data

    def cat_multi_encode(self,data, var, regroup_dict):
        """
        encode categorical variables with many categories, example: regroup_dict = {'new_cat':[old_cat,...]}
        first split , then encode
        :param data:
        :param var:
        :return:
        """
        new_data = data.fillna(value={var: 'NA'})
        # dict_encode = dict(zip(regroup_dict.keys(), range(len(regroup_dict))))
        result_data = new_data.apply(self.cat_multi_encode_onerow,axis=1,var=var,regroup_dict=regroup_dict)
        new_data.drop([var], axis=1,inplace=True)
        new_data.reset_index(drop=True, inplace=True)
        result_data.reset_index(drop=True, inplace=True)
        result_data = pd.concat([new_data, result_data],axis=1)
        return result_data


    def num_encode(self, data, var, proper_range=None):
        """
        attention: we replace the values of the variable and probably add a column indicate if it is NA
        :param data:
        :param var:
        :param proper_range:
        :return:
        """
        new_data = data.copy()
        new_data[var+'_NA'] = new_data[var].isnull().apply(lambda x: x*1)
        new_data = new_data.fillna(value={var: 0})
        # filter the values outside the proper range
        if proper_range is not None:
            new_data[var] = new_data[var].clip(proper_range[0], proper_range[1])
        return new_data

    def NL_encode(self,data,var,max_feature):
        """
        fill NAN with string 'NA', delete the original variable and add new variables representing term-document matrix
        :param data:
        :param var:
        :param max_feature:
        :return:
        """
        new_data = data.fillna(value={var: 'NA'})
        pattern = "[A-Z]"
        new_data_preprocessing = new_data[var].apply(self.check_content_NL,pattern=pattern)

        term_document_matrix, vectorizer_tfidf = self.var_vectorize(
            data=new_data_preprocessing,vectorizer=TfidfVectorizer(max_features=max_feature,
                                                                   strip_accents='unicode',
                                                                   stop_words=stopwords.words('french')
                                                                   ))
        # assert term_document_matrix.shape[1] == max_feature, 'max_feature does not equals to term_document_matrix.shape[1]'
        # replace the original column with the new NL columns, the number of columns depends on 'max_feature'
        new_data.drop([var], axis=1,inplace=True)
        feature_namelst = vectorizer_tfidf.get_feature_names()
        varlst = [var+'_'+ele for ele in feature_namelst]
        df_varlst= pd.DataFrame(term_document_matrix.toarray(),columns=varlst)
        new_data.reset_index(drop=True, inplace=True)
        df_varlst.reset_index(drop=True, inplace=True)
        new_data = pd.concat([new_data, df_varlst],axis=1)

        return new_data

    def check_content_NL(self, inner_data, pattern):
        # data : a string
        # replace the punctuation with space
        inner_data_str = str(inner_data)
        inner_new = inner_data_str.translate(str.maketrans(string.punctuation+'0123456789', ' ' * (len(string.punctuation)+10)))
        word_lst = inner_new.split()
        if any(i.isupper() for i in word_lst):
            return ' '.join([ele.lower() for ele in word_lst])
        else:
            # detect the word with uppercase beginning
            new_string = re.sub(pattern, lambda x: " " + x.group(0), inner_new).lower()
            # remove the space at the beginning and the bottom
            return ' '.join(new_string.split())

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

            data_tp = data_copy[col_lst_encode]

            array_insert = np.squeeze(data_tp.values.reshape(1,-1))
        # insert values to empty df
        #     print('len{}_{}'.format(len(array_insert),array_insert))
            new_data.ix[0, 0:len(array_insert)] = array_insert
            return new_data

    def merge_EqtInt_onerow(self,data_Eqt,data_Int,max_Int=3,Int_mergelst=Int_MERGE):
        # column names
        col_names = []
        for i in range(max_Int):
            lst_tp = [var+'_{}'.format(i) for var in Int_mergelst]
            col_names.extend(lst_tp)

        result_data = pd.DataFrame(columns=col_names)
        # pick the intervention records of this eqt
        data_pick = data_Int.loc[
            (data_Int['pan_CodeEqt'] == data_Eqt['eq_Code']) & (data_Int['region'] == data_Eqt['region'])]\
            .sort_values(by= 'int_Fin',na_position='first')
        data_pick = data_pick[Int_mergelst]

        if data_pick is None:
            pass
        elif len(data_pick) >= max_Int:
            one_row_df = data_pick.iloc[:max_Int].stack(dropna=False).to_frame().T
            one_row_df.columns = col_names
            # print('length >= 3')
            # print(one_row_df)
            result_data.loc[0] = one_row_df.loc[0]
        else:
            # fillna ='NA'
            # print(data_pick)
            one_row_df = data_pick.stack(dropna=False).to_frame().T
            # print('================')
            # print(len(one_row_df.columns))
            one_row_df.columns = col_names[:len(one_row_df.columns)]
            # print("=========================")
            # print('length < 3')
            # print(one_row_df)
            result_data.loc[0] = one_row_df.loc[0]
        result_data = pd.concat([data_Eqt,result_data.iloc[0]])
        # result_data = result_data.T
        # print(type(result_data.iloc[0]))
        return result_data

    def merge_EqtInt(self,data_Eqt,data_Int,max_Int=3,Int_mergelst=Int_MERGE):
        # simply merge
        result = data_Eqt.apply(func=self.merge_EqtInt_onerow,axis=1,data_Int=data_Int,max_Int=max_Int,Int_mergelst=Int_mergelst)
        result.drop_duplicates(inplace=True)
        return result

    def create_newvar(self,data_merge,max_Int=3):
        """
        this function is note generalized, only for create delai, create target
        :param data_merge:
        :return:
        """
        new_data = data_merge.copy()
        for i in range(max_Int):
            new_data['DelaiInt_{}'.format(i)] = new_data['int_DateIntervention_{}'.format(i)] - new_data['int_Fin_{}'.format(i)]

        for j in range(1,max_Int):
            new_data['PanneDelai_{}'.format(j)] = new_data['int_Fin_{}'.format(j)] - new_data['pan_DateSignal_{}'.format(j-1)]

        return new_data

