from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def generar_dataset_facil(n_samples=500, random_state=42):
    X, y = make_classification(n_samples=n_samples, n_features=2, 
                               n_informative=2, n_redundant=0, 
                               n_clusters_per_class=2, class_sep=2.0, 
                               random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)
    
    return X_train, X_test, y_train, y_test

def generar_dataset_dificil(n_samples=500, random_state=15):
    X, y = make_classification(n_samples=n_samples, n_features=2, 
                               n_informative=2, n_redundant=0, 
                               n_clusters_per_class=2, class_sep=1, 
                               flip_y=0.1, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)
    return X_train, X_test, y_train, y_test

