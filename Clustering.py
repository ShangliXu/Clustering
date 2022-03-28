# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:13:45 2022

@author: passi
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
# from scipy.stats import kurtosis
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import VarianceThreshold


random_seed = 42
impute_strategy = 'mean'
test_data_size = 0.2


def data_preprocess(X, y):
    col_names = X.columns
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    imputer = SimpleImputer(missing_values = np.nan, strategy = impute_strategy)
    imputer = imputer.fit(X)
    X = pd.DataFrame(data = imputer.transform(X), index = y.index, columns = col_names)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size = test_data_size, random_state = random_seed)
    return X_train, X_test, y_train, y_test


def get_kepler_data():
    kepler = pd.read_csv("Kepler Exoplanet Search Results.csv", index_col = 'kepid')
    col_to_drop = ['rowid', 'kepoi_name', 'koi_pdisposition', 'koi_tce_delivname']
    col_to_drop.extend(kepler.columns[kepler.isnull().sum()>kepler.shape[0]*0.1].to_list())
    kepler.drop(col_to_drop, axis = 1, inplace = True)
    
    le = LabelEncoder()
    kepler.koi_disposition = le.fit_transform(kepler.koi_disposition)
    y = kepler['koi_disposition']
    print(y.unique(), le.inverse_transform(y.unique()))
    
    X = kepler.loc[:, kepler.columns != 'koi_disposition']
    return data_preprocess(X, y)


def get_maternal_data():
    maternal = pd.read_csv('Maternal Health Risk Data Set.csv')
    le = LabelEncoder()
    maternal.RiskLevel = le.fit_transform(maternal.RiskLevel)
    y = maternal.RiskLevel
    print(y.unique(), le.inverse_transform(y.unique()))
    X = maternal.loc[:, maternal.columns != 'RiskLevel']
    return data_preprocess(X, y)


# The Elbow Method
# function returns WSS score for k values from 1 to kmax
def calculate_WSS(points, kmax):
    sse = []
    for k in range(1, kmax+1):
        kmeans = KMeans(n_clusters = k, random_state=random_seed).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0
    
        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += sum([(points[i, j] - curr_center[j])**2 for j in range(len(points[0]))])
        sse.append(curr_sse)

    plt.plot(list(range(1, kmax+1)), sse)
    plt.title('WSS-vs-k')
    plt.xlabel('k')
    plt.ylabel('Silhouette')
    plt.show()
    return sse


# Silhouette score
def calculate_Silhouette(points, kmax):
    sil = []    
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, kmax+1):
        kmeans = KMeans(n_clusters = k, random_state=random_seed).fit(points)
        labels = kmeans.labels_
        sil.append(silhouette_score(points, labels, metric = 'euclidean'))    
      
    plt.plot(list(range(2, kmax+1)), sil)
    plt.title('Silhouette-vs-k')
    plt.xlabel('k')
    plt.ylabel('Silhouette')
    plt.show()
    return sil
  


# Silhouette score
def EM_calculate_Silhouette(points, kmax):
    sil = []    
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, kmax+1):
        gm = GaussianMixture(n_components=k, random_state=random_seed).fit(points)
        labels = gm.predict(points)
        sil.append(silhouette_score(points, labels, metric = 'euclidean'))    
      
    plt.plot(list(range(2, kmax+1)), sil)
    plt.title('Silhouette-vs-k')
    plt.xlabel('k')
    plt.ylabel('WSS')
    plt.show()
    return sil  


def PCA_reduction(points):
    #95% of variance
    pca_transformer = PCA(n_components = 0.95)
    pca_transformer.fit(points)
    pca_points = pca_transformer.fit_transform(points)
    return pca_points, pca_transformer


def PCA_plot(points):
    pca = PCA().fit(points)

    plt.rcParams["figure.figsize"] = (12,6)
    
    fig, ax = plt.subplots()
    xi = np.arange(1, points.shape[1] + 1, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)
    
    plt.ylim(0.0,1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')
    
    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, 11, step=1)) #change from 0-based array index to 1-based human-readable label
    plt.ylabel('Cumulative variance (%)')
    plt.title('The number of components needed to explain variance')
    
    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)
    
    ax.grid(axis='x')
    plt.show()


def kurtos(x):
    n = np.shape(x)[0]
    mean = np.sum((x**1)/n) # Calculate the mean
    var = np.sum((x-mean)**2)/n # Calculate the variance
    skew = np.sum((x-mean)**3)/n # Calculate the skewness
    kurt = np.sum((x-mean)**4)/n # Calculate the kurtosis
    kurt = kurt/(var**2)-3

    return kurt, skew, var, mean


def ICA(points, kmax):
    kurt = []
    for k in range(1, kmax+1):
        transformer = FastICA(n_components=k, random_state=random_seed)
        X_transformed = transformer.fit_transform(points)
        kurt.append(np.mean(kurtos(X_transformed)))
        
    plt.plot(list(range(1, kmax+1)), kurt)
    plt.title('kurtosis-vs-k')
    plt.xlabel('k')
    plt.ylabel('kurt')
    plt.show()
    return kurt


def ICA_reduction(points, k):
    #95% of variance
    ica_transformer = FastICA(n_components=k, random_state=random_seed)
    ica_points = ica_transformer.fit_transform(points)
    return ica_points, ica_transformer


def RP(points, kmax):
    def inverse_transform_rp(rp, X_transformed, points):
        return X_transformed.dot(rp.components_) + np.mean(points, axis = 0)
    
    reconstruction_error = []
    for k in range(1, kmax+1):    
    	rp = GaussianRandomProjection(n_components = k, random_state=random_seed)
    	X_transformed = rp.fit_transform(points)
    	X_projected = inverse_transform_rp(rp, X_transformed, points)
    	reconstruction_error.append(((points - X_projected) ** 2).mean())
    
    plt.plot(list(range(1, kmax+1)), reconstruction_error)
    plt.title('reconstruction_error-vs-k')
    plt.xlabel('k')
    plt.ylabel('reconstruction_error')
    plt.show()
    return reconstruction_error



def RP_reduction(points, k):
    #95% of variance
    rp_transformer = GaussianRandomProjection(n_components=k, random_state=random_seed)
    rp_points = rp_transformer.fit_transform(points)
    return rp_points, rp_transformer



def VT_reduction(points):
    vt_transformer = VarianceThreshold(threshold=(.95 * (1 - .95)))
    vt_points = vt_transformer.fit_transform(points)
    return vt_points, vt_transformer




print('Kepler data')
X_train, X_test, y_train, y_test = get_kepler_data()
points = X_train.to_numpy()

kmax = 50
sse = calculate_WSS(points, kmax)
sil = calculate_Silhouette(points, kmax)
EM_sil = EM_calculate_Silhouette(points, kmax)

PCA_plot(points)
pca_points, pca_transformer = PCA_reduction(points)
pca_k = 5

kmax = 40
kurt = ICA(points, kmax)
ica_k = 36
ica_points, ica_transformer = ICA_reduction(points, ica_k)


reconstruction_error = RP(points, kmax)
rp_k = 13
rp_points, rp_transformer = RP_reduction(points, rp_k)


vt_points, vt_transformer = VT_reduction(points)




kmax = 40
for data in [pca_points, ica_points, rp_points, vt_points]:
    sse = calculate_WSS(data, kmax)
    sil = calculate_Silhouette(data, kmax)
    EM_sil = EM_calculate_Silhouette(data, kmax)











print('Health data')

X_train, X_test, y_train, y_test = get_maternal_data()
points = X_train.to_numpy()

kmax = 6
sse = calculate_WSS(points, kmax)
sil = calculate_Silhouette(points, kmax)
EM_sil = EM_calculate_Silhouette(points, kmax)

PCA_plot(points)
pca_points, pca_transformer = PCA_reduction(points)
pca_k = 5


kurt = ICA(points, kmax)
ica_k = 1
ica_points, ica_transformer = ICA_reduction(points, ica_k)


reconstruction_error = RP(points, kmax)
rp_k = 5
rp_points, rp_transformer = RP_reduction(points, rp_k)


vt_points, vt_transformer = VT_reduction(points)




for data in [pca_points, ica_points, rp_points, vt_points]:
    sse = calculate_WSS(data, kmax)
    sil = calculate_Silhouette(data, kmax)
    EM_sil = EM_calculate_Silhouette(data, kmax)
    
    
    
    
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 16:24:42 2022

@author: passi
"""
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ShuffleSplit, GridSearchCV, learning_curve
from sklearn.metrics import balanced_accuracy_score
# from sklearn.pipeline import make_pipeline

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

random_seed = 42
test_data_size = 0.2
cv_data_size = 0.2
impute_strategy = 'mean'
n_jobs_val = -1 
cv_num = 5
max_iter_num = 200
score_method = 'balanced_accuracy' #balanced_accuracy_score
cv_splitter = ShuffleSplit(n_splits = cv_num, test_size = cv_data_size, random_state = random_seed)
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

estimator_names = ['Neural networks']
iterative_algorithms = ['Neural networks']
estimator_list = [MLPClassifier]
estimators = dict(zip(estimator_names, estimator_list))
    
hyperparameter_list = [
    {'hidden_layer_sizes': [(50,50,50), (100,)], 'max_iter': [50, 100, 200, 500],
    'alpha': [0.0001, 0.05], 'learning_rate': ['constant','adaptive'],}]
hyperparameters = dict(zip(estimator_names, hyperparameter_list))

class supervised_learning:
    def __init__(self, estimator_names):
        self.estimator_names = estimator_names
        self.best_hyperparameters = {}

    
    def tuning_hyperparameter(self, X_train, X_test, y_train, y_test, estimator_name): 
        self.estimator_name = estimator_name
        estimator = estimators[estimator_name]
        parameter_space = hyperparameters[estimator_name]
        if estimator_name != 'k-nearest neighbors':
            clf_func = estimator(random_state = random_seed)
        else:
            clf_func = estimator()
        # higher the AUC value for a classifier, the better its ability to distinguish
        clf = GridSearchCV(clf_func, parameter_space, n_jobs = n_jobs_val, cv = cv_splitter, scoring = score_method, return_train_score = True)
        clf.fit(X_train, y_train)

        params = clf.cv_results_['params']
        train_means = clf.cv_results_['mean_train_score']
        test_means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        cv_results = pd.concat([pd.DataFrame(params), pd.DataFrame(list(zip(train_means, test_means, stds)), \
                                            columns =['train_score', 'test_score', 'std_test_score'])], axis = 1)
        cv_results['diff_score'] = abs(cv_results.train_score - cv_results.test_score)
        self_defined_best_params = clf.cv_results_['params'][cv_results.diff_score[cv_results.test_score>max(cv_results.test_score)*0.99].idxmin()]
        cv_results.to_csv(estimator_name + '.csv')
        
        plt.figure(figsize=(20, 6))
        plt.scatter(list(map(str, params)), train_means, label = 'train_score')
        plt.scatter(list(map(str, params)), test_means, label = 'test_score')
        plt.axvline(x = str(self_defined_best_params), ls = '--', label = 'best_param')
        plt.title(self.estimator_name + ' Hyperparameter tuning results', fontsize = 15)
        plt.ylabel('score', fontsize = 12)
        plt.xticks(rotation = 45, ha = 'right', fontsize = 12)
        plt.legend()
        plt.plot()
        
        self.estimator, self.best_params = estimator, self_defined_best_params#clf.best_params_
        return self.estimator, self.best_params #estimator(random_state = random_seed, **clf.best_params_)
    
        
    def plot_learning_curve(self, X_train, y_train, train_sizes = np.linspace(0.1, 1.0, 5),):
        if estimator_name != 'k-nearest neighbors':
            best_estimator = self.estimator(random_state = random_seed, **self.best_params)
        else:
            best_estimator = self.estimator(**self.best_params)
    
        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            best_estimator, X_train, y_train, cv = cv_splitter, n_jobs = n_jobs_val,
            train_sizes = train_sizes, return_times = True, scoring = score_method)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)
    
        # Plot learning curve
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - 2 * train_scores_std,
            train_scores_mean + 2 * train_scores_std, alpha=0.1, color="r",)
        axes[0].fill_between(train_sizes, test_scores_mean - 2 * test_scores_std,
            test_scores_mean + 2 * test_scores_std, alpha=0.1, color="g", )
        axes[0].plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")
        axes[0].legend(loc="best")
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")    
        axes[0].set_title(self.estimator_name + " learning curve")
    
        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, "o-")
        axes[1].fill_between(train_sizes, fit_times_mean - 2 * fit_times_std,
            fit_times_mean + 2 * fit_times_std, alpha=0.1,)
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title(self.estimator_name + " model scalability")
    
        # Plot fit_time vs score
        fit_time_argsort = fit_times_mean.argsort()
        fit_time_sorted = fit_times_mean[fit_time_argsort]
        test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
        test_scores_std_sorted = test_scores_std[fit_time_argsort]
        axes[2].grid()
        axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
        axes[2].fill_between(fit_time_sorted, test_scores_mean_sorted - 2 * test_scores_std_sorted,
            test_scores_mean_sorted + 2 * test_scores_std_sorted, alpha=0.1,)
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title(self.estimator_name + " model performance")

        plt.show()
        return
    
    
    @ignore_warnings(category=ConvergenceWarning)
    def plot_learning_curve_on_iteration(self, X_train, y_train):
        max_iter_num = best_params['max_iter']
        epochs = range(max_iter_num) if max_iter_num < 500 else range(0, max_iter_num + max_iter_num//10, max_iter_num//10)
        
        def training_cv_score(best_estimator, training_score_list, validation_score_list):
            y_train_pred = best_estimator.predict(x_training)
            # Multi-layer Perceptron classifier optimizes the log-loss function using LBFGS or stochastic gradient descent.
            curr_train_score = balanced_accuracy_score(y_training, y_train_pred) # training performances
            y_val_pred = best_estimator.predict(x_validation) 
            curr_valid_score = balanced_accuracy_score(y_validation, y_val_pred) # validation performances
            training_score_list.append(curr_train_score) # list of training perf to plot
            validation_score_list.append(curr_valid_score) # list of valid perf to plot
            
        training_score, validation_score = [], []        
        for train_index, test_index in cv_splitter.split(X_train):
            x_training, x_validation = X_train.iloc[train_index], X_train.iloc[test_index]
            y_training, y_validation = y_train.iloc[train_index], y_train.iloc[test_index]
            training_score_list, validation_score_list = [], []            
           
            if self.estimator_name == 'Neural networks':
                best_estimator = self.estimator(random_state = random_seed, **self.best_params)
                best_estimator.warm_start = True
                best_estimator.max_iter = 1 
                for epoch in epochs:       
                    best_estimator.partial_fit(x_training, y_training, classes = np.unique(y_training))                     
                    training_cv_score(best_estimator, training_score_list, validation_score_list)

            elif self.estimator_name == 'Support Vector Machines':
                for epoch in epochs:
                    best_estimator = self.estimator(random_state = random_seed, **self.best_params)
                    best_estimator.max_iter = epoch
                    best_estimator.fit(x_training, y_training)
                    training_cv_score(best_estimator, training_score_list, validation_score_list)
                    
            training_score.append(training_score_list)
            validation_score.append(validation_score_list)                    
                    
        plt.plot(epochs, np.array(training_score).mean(axis = 0), label = 'Training')
        plt.plot(epochs, np.array(validation_score).mean(axis = 0), label = 'Cross-validation')
        plt.title(self.estimator_name + ' Accuracy under cross validation')
        plt.xlabel('Iterations')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()
    
        return training_score, validation_score
        
        
    def predict_test_data(self, X_train, X_test, y_train, y_test):
        if self.estimator_name == 'Neural networks':
            best_estimator = self.estimator(random_state = random_seed, **self.best_params)
        else:
            best_estimator = self.estimator(**self.best_params)
        best_estimator.fit(X_train, y_train)
        y_pred = best_estimator.predict(X_test)
        test_score = balanced_accuracy_score(y_test, y_pred)
        return test_score



sl = supervised_learning(estimator_names)
out_of_sample_scores = []
# for get_data in [sl.get_kepler_data, sl.get_maternal_data]:   
#     X_train, X_test, y_train, y_test = get_data()  
for X_train, transformer in zip([pca_points, ica_points, rp_points, vt_points], [pca_transformer, ica_transformer, rp_transformer, vt_transformer]):
    X_test = transformer.fit_transform(X_test)
    X_train, X_test = pd.DataFrame(X_train), pd.DataFrame(X_test)
    out_of_sample_score = []
    for estimator_name in sl.estimator_names:
        estimator, best_params = sl.tuning_hyperparameter(X_train, X_test, y_train, y_test, estimator_name)
        sl.best_hyperparameters[estimator_name] = best_params
        sl.plot_learning_curve(X_train, y_train)
        if estimator_name in iterative_algorithms:
            training_score, validation_score = sl.plot_learning_curve_on_iteration(X_train, y_train)
        test_score = sl.predict_test_data(X_train, X_test, y_train, y_test)
        out_of_sample_score.append(test_score)
    out_of_sample_scores.append(out_of_sample_score)

for i, data in enumerate(['kepler', 'health']):
    plt.plot(estimator_names, out_of_sample_scores[i])
    plt.title(data + ' Score on testing data')
    plt.ylabel('Score')
    plt.xticks(rotation = 15, ha = 'right')
    plt.show()