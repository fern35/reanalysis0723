import pandas as pd

class Cleaner(object):

    def rv_dupRow(self,data,Var_lst=None):
        """remove duplicated rows"""
        new_data = data.drop_duplicates(subset=Var_lst)
        return new_data

    def rv_redunVar(self,data):
        """remove the NAN variables and record the number of redundant variables"""
        # remove the columns with all equal NAN
        new_data = data.dropna(axis=1, how='all', inplace=True)
        # remove the columns with all the same values
        new_data = new_data.ix[:, (new_data != new_data.ix[0]).any()]
        # len_redun = len(data.columns) - len(new_data.columns)
        return new_data

    def rep_NA(self,data):
        return data.fillna('blank')

    def add_dur(self,data,Var_lst,currtime):
        """
        :param data: dataframe
        :param Var_lst: list of variables
        :param currtime: datetime which indicates the referencing datetime
        :return: the dataframe with new variables which indicate the duration
        """
        # inner function for return the days
        new_data = data.copy()
        def return_days(a):
            return a.days

        for var in Var_lst:
            new_var = (currtime - pd.to_datetime(data[var])).apply(return_days)
            new_data[var+'_Duree'] = new_var

        return new_data

    def rep_dur(self,data,Var_lst,currtime):
        """
        :param data: dataframe
        :param Var_lst: list of variables
        :param currtime: datetime which indicates the referencing datetime
        :return: the dataframe replaced with new variables which indicate the duration
        """
        # inner function for return the days
        new_data = data.copy()
        def return_days(a):
            return a.days

        for var in Var_lst:
            new_var = (currtime - pd.to_datetime(data[var])).apply(return_days)
            new_data[var] = new_var

        return new_data

