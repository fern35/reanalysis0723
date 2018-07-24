# Check the MUT CAT vars and NL vars, find proper methods for processing and generate related constants
from modules.analyser import Analyzer
from utils.constants import VILLE_NAME, Armoire_PICK, Int_PICK, PL_PICK

date_str = '0723'
analyzer = Analyzer(datestr=date_str)

"""
MUL CAT vars
PL: lampe_Type
INT: pan_Solde, int_Solde, int_ElemDefaut, int_TypeTnt, int_TypeEqt, pan_TypeEqt, pan_Defaut, Elem_Defaut
"""
# generate doc for analyzing the MUL CAT vars
analyzer.comp_Var_cities(foldername='Armoire',villelst=VILLE_NAME,group_dict= Armoire_PICK)
analyzer.comp_Var_cities(foldername='Int',villelst=VILLE_NAME,group_dict= Int_PICK)
analyzer.comp_Var_cities(foldername='PL',villelst=VILLE_NAME,group_dict= PL_PICK)
# Probs:
# 1. int_Solde & pan_Solde, int_Defaut & pan_Defaut, int_TypeEqt & pan_TypeEqt should be the same
# 2. merge the categories:
# * pan_Solde, int_Solde completed
