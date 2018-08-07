from datetime import datetime
import os
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram

class Plotter(object):
    """docstring for Plotter"""

    def __init__(self,datestr,savedir = None):
        rcParams.update({'figure.autolayout': True})
        if savedir is None:
            self.savedir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..','..','data_save{}'.format(datestr),'img')
        else:
            self.savedir = savedir
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)

    def check_savepath(self,foldpath, filename):
        if not os.path.exists(foldpath):
            os.makedirs(foldpath)
        save_path = os.path.join(foldpath,filename)
        try:
            os.remove(save_path)
        except OSError:
            pass
        return save_path

    def plot_elbow(self,hcluster,foldername,title):
        last = hcluster[-40:, 2]
        last_rev = last[::-1]
        idxs = np.arange(1, len(last) + 1)
        # plt.figure(figsize=(25, 10))
        fig = plt.figure()
        plt.title(title)
        plt.plot(idxs, last_rev)
        plt.xlabel('no of clusters')
        plt.ylabel('distance')
        plt.show()
        fig.savefig(os.path.join(
            self.savedir, foldername,'{}.png'.format(title)))

    def plot_dendro(self,hcluster,foldername,title):
        fig=plt.figure(figsize=(25, 10))
        plt.title(title)
        plt.xlabel('sample index')
        plt.ylabel('distance')
        dendrogram(
            hcluster,
            leaf_rotation=90.,
            leaf_font_size=12,
            # count_sort=True,
            # distance_sort=True,
            show_leaf_counts=True
        )
        plt.show()
        fold_path = os.path.join(
            self.savedir,foldername)
        save_path = self.check_savepath(foldpath=fold_path,filename='{}.png'.format(title))

        fig.savefig(save_path)

