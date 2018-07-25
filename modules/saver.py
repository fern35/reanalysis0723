import os
import pandas as pd

class Saver(object):
    """docstring for Loader"""
    def __init__(self, datestr,datasavedir=""):
        if datasavedir == "":
            self.basedir = os.path.abspath(os.path.dirname(__file__))
            self.datasavedir = os.path.abspath(
                os.path.join(self.basedir, '..','..','data_save{}'.format(datestr)))
        else:
            self.datasavedir = datasavedir

    def check_savepath(self,foldpath, filename):
        if not os.path.exists(foldpath):
            os.makedirs(foldpath)
        save_path = os.path.join(foldpath,filename)
        try:
            os.remove(save_path)
        except OSError:
            pass
        return save_path

    def save_excel(self, data,filename,foldername=''):
        """save dataframe to excel"""
        fold_path = os.path.join(self.datasavedir,'excel','{}'.format(foldername))
        save_path = self.check_savepath(foldpath=fold_path,filename='{}.xlsx'.format(filename))
        writer = pd.ExcelWriter(save_path)
        data.to_excel(writer, 'Sheet1')
        writer.save()
        return data

