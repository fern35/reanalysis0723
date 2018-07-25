import os
import pandas as pd

class Loader(object):
    """docstring for Loader"""

    def __init__(self, datestr,datadir=""):
        if datadir == "":
            self.basedir = os.path.abspath(os.path.dirname(__file__))
            self.datadir = os.path.join(self.basedir, '..','..', 'data')
            self.datasavedir = os.path.join(self.basedir, '..','..', 'data_save{}'.format(datestr))
        else:
            self.datadir = datadir

    def load_ArmPL(self, foldername,filename, NAME_LIST):
        """
        load the excel BDD Armoire/PL of one city as dataframe
        :param foldername:
        :param filename:
        :param NAME_LIST:
        :return:
        """
        data = pd.read_excel(os.path.join(self.datadir, foldername,filename), skiprows=[0])
        # change the names of the variables
        data.columns = NAME_LIST
        return data

    def load_Intervention(self, foldername,filename_lst, NAME_LIST):
        """
        load the excel BDD Intervention of one city as dataframe
        :param foldername:
        :param filename_lst:
        :param NAME_LIST:
        :return:
        """
        data_Int = pd.DataFrame(columns=NAME_LIST)
        for file_name in filename_lst:
            data_Int_tp = pd.read_excel(os.path.abspath(os.path.join(self.datadir,foldername,'Intervention' ,file_name)), skiprows=[0])
            data_Int_tp.drop(columns=[data_Int_tp.columns[0]], inplace=True)
            data_Int_tp.columns = NAME_LIST
            data_Int = data_Int.append(data_Int_tp, ignore_index=True)
        data_Int.dropna(axis=0, how='all',inplace=True)
        return data_Int

    def load_excel(self,foldername,filename):
        """

        :param foldername:
        :param filename:
        :return:
        """
        excel_path = os.path.join(self.datasavedir, 'excel')
        data = pd.read_excel(os.path.join(excel_path, foldername,filename+'.xlsx'))
        return data