#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import graphviz
from sklearn import tree
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# ### Fonctions arbre de décision

# #### Fonction de préparation des données

# In[6]:


def prepare_data(df, var_x, var_y, test_size=0.3, random_state=42):
    """
    Prépare les données en séparant les variables indépendantes et dépendantes,
    effectue un train-test split, et applique SMOTE pour équilibrer les classes.
    """
    X, y = df[var_x], df[var_y]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, y_train = SMOTE().fit_resample(X_train, y_train)  # Appliquer SMOTE pour équilibrer les classes
    return X_train, X_test, y_train, y_test


# #### Fonction de cross-validation

# In[7]:


# Fonction de validation croisée pour l'optimisation bayésienne
def dtree_cv(max_depth, min_samples_split, min_samples_leaf, X_train, y_train):
    """
    Effectue une validation croisée pour évaluer l'impact des hyperparamètres de l'arbre de décision
    """
    estimator = DecisionTreeClassifier(
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        random_state=42
    )
    cval = cross_val_score(estimator, X_train, y_train, scoring='f1', cv=5)
    return cval.mean()  # Retourne la moyenne des scores F1


# #### Fonction pour appliquer l'optimisation bayésienne

# In[8]:


def optimize_bayesian(param_bounds, X_train, y_train):
    """
    Optimisation bayésienne des hyperparamètres d'un arbre de décision.
    """
    # Initialiser l'optimiseur bayésien
    optimizer = BayesianOptimization(
        f=lambda max_depth, min_samples_split, min_samples_leaf: dtree_cv(
            max_depth, min_samples_split, min_samples_leaf, X_train, y_train),
        pbounds=param_bounds,
        random_state=1,
    )

    # Lancer l'optimisation
    optimizer.maximize(n_iter=15, init_points=5)  # Optimisation avec n_iter itérations et init_points points initiaux

    # Obtenir les meilleurs paramètres
    best_params_bayes = optimizer.max['params']
    best_params_bayes['max_depth'] = int(best_params_bayes['max_depth'])
    best_params_bayes['min_samples_split'] = int(best_params_bayes['min_samples_split'])
    best_params_bayes['min_samples_leaf'] = int(best_params_bayes['min_samples_leaf'])
    best_score_bayes = optimizer.max['target']

    print(f"Best Parameters (Bayesian Optimization): {best_params_bayes}")
    print(f"Best Score (Bayesian Optimization): {best_score_bayes}")
    
    return best_params_bayes


# #### Fonction d'entraînement d'arbre de décision

# In[9]:


# Entraîner un modèle d'arbre de décision avec les meilleurs hyperparamètres
def train_final_model(X_train, y_train, best_params_bayes):
    """
    Entraîne un modèle d'arbre de décision avec les meilleurs hyperparamètres.
    """
    final_model = DecisionTreeClassifier(
        max_depth=best_params_bayes['max_depth'],
        min_samples_split=best_params_bayes['min_samples_split'],
        min_samples_leaf=best_params_bayes['min_samples_leaf'],
        random_state=42
    )
    final_model.fit(X_train, y_train)
    return final_model


# #### Fonction d'affichage de l'arbre de décision

# In[10]:


def graph_clf(model, X_train):
    """
    Génère et affiche un graphique de l'arbre de décision.
    """
    dot_data = tree.export_graphviz(model, out_file=None,
                                    feature_names=X_train.columns,
                                    filled=True, rounded=True,
                                    special_characters=True,
                                   class_names=True)
    return dot_data
# graphviz.Source(dot_data) pour visualiser l'arbre de décision


# #### Fonction pour afficher la matrice de confusion

# In[11]:


# Fonction pour afficher la matrice de confusion
def plot_conf_mat(y_test, y_pred, normalize=None):
    """
    Affiche la matrice de confusion pour évaluer la performance du modèle.
    """
    cm = confusion_matrix(y_test, y_pred, normalize=normalize)
    disp = ConfusionMatrixDisplay(cm)
    
    # Affichage de la matrice de confusion
    fig, ax = plt.subplots()
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    
    # Enlever la grille
    ax.grid(False)  # Désactive la grille
    
    plt.show()


# #### Fonction principale

# In[12]:


def main(df, var_x, var_y, param_bounds):
    """
    Protocole complet pour l'optimisation bayésienne et l'évaluation du modèle.
    Cette version retourne des objets utiles pour une utilisation ultérieure et affiche les meilleurs hyperparamètres.
    """
    # Préparer les données
    X_train, X_test, y_train, y_test = prepare_data(df, var_x, var_y)

    # Optimiser les hyperparamètres
    best_params_bayes = optimize_bayesian(param_bounds, X_train, y_train)

    # Entraîner le modèle final
    final_model = train_final_model(X_train, y_train, best_params_bayes)

    # Faire des prédictions sur le jeu de test
    y_pred = final_model.predict(X_test)

    # Évaluation : matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    
    # Affichage de l'arbre de décision
    graph = graph_clf(final_model, X_train)

    # Créer le dictionnaire avec tous les résultats
    output = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'best_params_bayes': best_params_bayes,
    'final_model': final_model,
    'y_pred': y_pred,
    'graph': graph,
    'confusion_matrix': cm
    }

    # Retourner le dictionnaire contenant tout
    return output

