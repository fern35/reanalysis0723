# list of cities
VILLE_NAME = ['CAVB','BESSANCOURT',
              # 'BOGOR',
              'GOUSSAINVILLE','GUADELOUPE','MONTESSON','NOUMEA']

# all the variables of Armoire

Armoire_COORDONNEE  = ['coor_X', 'coor_Y', 'coor_Projection']
Armoire_EQUIPEMENT = ['eq_Type', 'eq_Code', 'eq_Alias', 'eq_NoVoie', 'eq_Commune', 'eq_TypeVoie', 'eq_NomVoie',
                   'eq_Quartier', 'eq_DateInstallation', 'eq_DateDernierMaintenance', 'eq_Vetuste', 'eq_Marque',
                   'eq_EtatExploitation', 'eq_DateCreation', 'eq_ActeurCreation', 'eq_NombrePhoto', 'eq_Commentaire']
Armoire_ArmoireBT = ['arm_NoDepart', 'arm_NoLampe', 'arm_Reseau', 'arm_TypeSerrure', 'arm_Variateur', 'arm_Dimension',
                   'arm_TypeFixation', 'arm_TypeEmplacement', 'arm_NoCompteur', 'arm_TypeCompteur',
                   'arm_Livraison',
                   'arm_Tension','arm_TypeAlimentation', 'arm_Aperage', 'arm_PuissanceSouscrite', 'arm_TypeTarif', 'TypeAllumage',
                   'arm_DureeFonctionnement', 'arm_MiseTerre', 'arm_Consignation']
Armoire_MODELE = ['modArm_nomModele', 'modArm_DureeVie', 'modArm_Materiau', 'modArm_NoPorte']
Armoire_DEPART = ['depart_MarqueCellule', 'depart_ModeleCellule', 'depart_TypeProtection', 'depart_DateInstallation',
                   'depart_DernierMaintenance',
                   'depart_NoDepart', 'depart_TypeAlimentation', 'depart_Amperage',
                   'depart_Commentaire']

Armoire_GROUP = {'COORDONNEE':Armoire_COORDONNEE, 'EQUIPEMENT':Armoire_EQUIPEMENT,
                 'ArmoireBT':Armoire_ArmoireBT,'MODELE':Armoire_MODELE,'DEPART':Armoire_DEPART}

# Armoire_PICK = {'EQUIPEMENT':['eq_EtatExploitation','eq_Vetuste'],
#                  'ArmoireBT':['arm_Variateur', 'arm_NoLampe'],'DEPART':['depart_TypeAlimentation']}

Armoire_PICK = {'CAT':['eq_EtatExploitation','eq_Vetuste','arm_Variateur'],
                'DIST':['arm_NoLampe']}

Armoire_NAME = Armoire_COORDONNEE + Armoire_EQUIPEMENT + Armoire_ArmoireBT + Armoire_MODELE + Armoire_DEPART
# the variables which need to be replaced by the duration
Armoire_TIME = [ 'eq_DateDernierMaintenance','eq_DateCreation']

# the categorical variables which need to plot pie graph and belong to Armoire (the variables with no observations not included)
Armoire_ARM_CAT = ['eq_Vetuste','eq_EtatExploitation','arm_NoDepart',
               'arm_Reseau','arm_TypeSerrure','arm_Variateur','arm_TypeFixation',
               'arm_TypeEmplacement','arm_TypeCompteur','arm_Tension','arm_TypeAlimentation',
               'TypeAllumage','arm_DureeFonctionnement','arm_MiseTerre','arm_Consignation'
               ]

# thr variables which need to plot pie graph and only belong to Depart
Armoire_DEPART_CAT = ['depart_NoDepart', 'depart_TypeAlimentation', 'depart_TypeProtection', 'depart_Amperage']
# the variables which need distribution plots
Armoire_ARM_DIST = ['eq_DateDernierMaintenance','eq_DateCreation','arm_NoLampe']

Armoire_CLUSTERING = ['eq_Code','arm_NoLampe','eq_Vetuste','eq_Commentaire']

############################################################################################################################
# all the variables of Point Lumineux
PL_COORDONNEE = ['coor_X', 'coor_Y', 'coor_Projection']
PL_EQUIPEMENT = ['eq_Type', 'eq_Code', 'eq_TypeAmont', 'eq_CodeAmont', 'eq_TypeRacine', 'eq_CodeRacine', 'eq_Alias',
              'eq_NoVoie',
              'eq_Commune', 'eq_TypeVoie', 'eq_NomVoie', 'eq_Quartier', 'eq_DateInstallation',
              'eq_DateDernierMaintenance',
              'eq_EtatExploitation', 'eq_DateCreation', 'eq_ActeurCreation', 'eq_NombrePhoto', 'eq_Commentaire']
PL_PL = ['pl_TypeSDAL', 'pl_NoDepart', 'pl_Reseau', 'pl_NoLanterne']
PL_SUPPORT = ['sup_Type', 'sup_Code', 'sup_Marque', 'sup_Modele', 'sup_Forme', 'sup_Diametre',
              'sup_NoteCalcul', 'sup_Materiau', 'sup_Traitement', 'sup_Entraxe', 'sup_Vetuste', 'sup_MiseTerre',
              'sup_DateInstallation', 'sup_Couleur', 'sup_Arceau', 'sup_Hauteur', 'sup_CoffretRaccordement',
              'sup_EtatCoffret', 'sup_Commentaire']
PL_CONSOLE=['con_Marque', 'con_Modele', 'con_Code', 'con_Couleur', 'con_Angle', 'con_Avance', 'con_Forme',
              'con_DiametreLuminaire', 'con_Hauteur', 'con_HauteurParSol', 'con_Vetueste', 'con_Materiau',
              'con_Traitement', 'con_DateInstallation', 'con_Commentaire']
PL_LANTERNE = ['lan_Type', 'lan_Code', 'lan_Marque', 'lan_Modele', 'lan_Couleur', 'lan_Hauteur', 'lan_Fixation',
              'lan_Vasque', 'lan_Vetuste', 'lan_DateInstallation', 'lan_DateIntervention', 'lan_Ulor',
              'lan_Commentaire',
              # 'SL21_Code','SL21_Marque','SL21_Modele','SL21_DateInstallation','SL21_DateIntervention','SL21_NumBallast',
              # 'SL21_AbscenceDALI','SL21_Commentaire'
               ]
PL_APPEILLAGE = ['app_Code', 'app_Marque', 'app_Modele', 'app_DateInstallation', 'app_Emplacement', 'app_Type',
              'app_Gradation',
              'app_NiveauVariation', 'app_PlageVariation', 'app_Puissance', 'app_TypeSource', 'app_Vetuste',
              'app_Commentaire']
PL_LAMPE = ['lampe_Code', 'lampe_Marque', 'lampe_Modele', 'lampe_Forme', 'lampe_Couleur', 'lampe_IRC',
              'lampe_TypeCulot',
              'lampe_Puissance', 'lampe_Flux', 'lampe_Type', 'lampe_Vetuste', 'lampe_DureeVie', 'lampe_Tension',
              'lampe_DateInstallation', 'lampe_DateIntervention', 'lampe_Commentaire']

PL_GROUP = {'COORDONNEE':PL_COORDONNEE, 'EQUIPEMENT':PL_EQUIPEMENT,
            'PL':PL_PL,'SUPPORT':PL_SUPPORT,'CONSOLE':PL_CONSOLE,
            'LANTERNE':PL_LANTERNE,'APPEILLAGE':PL_APPEILLAGE,
            'LAMPE':PL_LAMPE}

# PL_PICK = {'EQUIPEMENT':['eq_EtatExploitation'],
#            'PL':['pl_Reseau', 'pl_NoLanterne'],
#            'SUPPORT':['sup_Vetuste', 'sup_Materiau', 'sup_Type'],
#             'LANTERNE':['lan_Vetuste'],'APPEILLAGE':['app_Puissance'],
#             'LAMPE':['lampe_Puissance']}

PL_PICK = {'CAT':['eq_EtatExploitation','pl_Reseau','sup_Vetuste', 'sup_Materiau', 'sup_Type',
                  'lan_Vetuste','lampe_Type'],'DIST':['pl_NoLanterne','app_Puissance','lampe_Puissance']}

PL_NAME = PL_COORDONNEE + PL_EQUIPEMENT + PL_PL + PL_SUPPORT + PL_CONSOLE + PL_LANTERNE + PL_APPEILLAGE + PL_LAMPE

PL_TIME = ['eq_DateInstallation','eq_DateDernierMaintenance','eq_DateCreation','sup_DateInstallation',
           'con_DateInstallation']
# the categorical variables which need to plot pie graph
PL_PL_CAT = ['eq_TypeAmont','eq_TypeRacine','eq_EtatExploitation','pl_Reseau','pl_NoLanterne',
         'sup_Type','sup_Materiau','sup_Traitement','sup_Entraxe','sup_Vetuste','sup_MiseTerre',
         'sup_Arceau','sup_CoffretRaccordement','sup_EtatCoffret','con_Marque','con_Modele',
         'con_Angle','con_Avance','con_Forme','con_DiametreLuminaire','con_Vetueste','con_Materiau'
         ]

PL_PL_DIST = ['eq_DateInstallation','eq_DateDernierMaintenance','eq_DateCreation','sup_DateInstallation',
           'con_DateInstallation','sup_Hauteur','con_Hauteur','con_HauteurParSol']

PL_LAN_CAT = ['lan_Type','lan_Marque', 'lan_Modele', 'lan_Couleur', 'lan_Fixation','lan_Vasque', 'lan_Vetuste',

         'app_Marque', 'app_Modele','app_Emplacement', 'app_Type','app_Gradation','app_NiveauVariation',
          'app_PlageVariation',  'app_TypeSource', 'app_Vetuste','lampe_Marque','lampe_Forme','lampe_Couleur',
          'lampe_TypeCulot','lampe_Type', 'lampe_Vetuste']

PL_LAN_DIST = ['app_Puissance', 'lan_Hauteur', 'lampe_Puissance', 'lampe_Tension']

PL_CLUSTERING = ['eq_Code', 'pl_Reseau',  'pl_NoLanterne', 'lan_Vetuste', 'lampe_Puissance', 'lampe_Type']

################################################################################################################################
# all the variables of Intervention
Int_PANNE = ['pan_Code', 'pan_DateSignal', 'pan_HeureSignal', 'pan_UtilisateurdDeclarant', 'pan_Declarant',
               'pan_CodeEqt',
               'pan_SourceEqt', 'pan_Ville', 'pan_Voie', 'pan_No', 'pan_TypeEqt', 'pan_Defaut', 'pan_NoPLimp',
               'pan_Solde', 'pan_Commentaire',
               'pan_DelaiIntCont', 'pan_DelaiInt', 'pan_MiseSecurite', 'pan_MiseProvisoire', 'pan_reparation',
               'pan_Astreinte', 'pan_Surtension',
               'pan_TypeControle']
Int_Intervention = ['int_DateIntervention', 'int_Debut', 'int_Fin', 'int_Constat', 'int_Defaut', 'int_NoPLimp',
               'int_CodeEqt', 'int_SourceEqt',
               'int_Ville', 'int_Voie', 'int_No', 'int_TypeEqt', 'int_ElemDefaut', 'int_CauseDefaut', 'int_TypeInt',
               'int_Chef', 'int_MoyenHumMob',
               'int_MatMob', 'int_Solde', 'int_Commentaire', 'int_TypeControle']
Int_GROUP = {'PANNE':Int_PANNE,'Intervention':Int_Intervention}
# Int_PICK = {'PANNE':['pan_DateSignal', 'pan_TypeEqt', 'pan_Solde',
#                      'pan_Commentaire',
                     # 'pan_Defaut', 'pan_DelaiInt','pan_NoPLimp'],'Intervention':['int_TypeEqt','int_Debut','int_DateIntervention','int_TypeInt', 'int_ElemDefaut']}

Int_PICK = {'CAT':['pan_TypeEqt','pan_Solde','pan_Defaut', 'int_Defaut','int_TypeEqt','int_ElemDefaut'],'DIST':['pan_DelaiInt','pan_NoPLimp','int_Debut','int_DateIntervention']}

Int_NAME = Int_PANNE + Int_Intervention
Int_TIME = ['pan_DateSignal','int_DateIntervention','int_Debut','int_Fin','int_NoPLimp', 'int_Fin']

Int_PAN_CAT = ['pan_TypeEqt','pan_Defaut','pan_Solde','pan_Astreinte']

Int_PAN_DIST = ['pan_DateSignal','pan_NoPLimp','pan_DelaiInt','pan_MiseSecurite','pan_MiseProvisoire',
                'pan_reparation',]

Int_INT_CAT = ['int_Constat','int_Defaut','int_TypeEqt','int_ElemDefaut','int_CauseDefaut','int_TypeInt',
               'int_MatMob','int_Solde']

Int_INT_DIST = ['int_DateIntervention','int_NoPLimp','int_MoyenHumMob']

Int_CLUSTERING = ['pan_Code', 'pan_CodeEqt', 'pan_DateSignal', 'pan_TypeEqt', 'pan_Solde', 'pan_Commentaire', 'pan_SourceEqt',
                  'pan_Defaut', 'pan_DelaiInt', 'int_DateIntervention', 'int_ElemDefaut', 'int_Fin', 'int_Solde',
                  'int_TypeInt', 'int_Defaut', 'int_Commentaire']