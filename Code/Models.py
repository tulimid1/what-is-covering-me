# import libraries 
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
plt.rcParams.update({'font.size': 16}) 
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
import time 
import seaborn as sns
import savingfigR as sf 

# KNN
class pKNN():
    # best parameters
    n_neighbors_best = 5
    weight_best = 'distance'
    # parameter options tested
    n_neighbors_options = np.arange(1, 15, 1)
    weights_options = ['uniform', 'distance']

    def __init__(self):
        # method calls 
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data(perform_scale=False, sub_data_section='')
        self.un_classifiers = np.unique(self.y_train)
        # Raw data 
        start = time.time()
        KNN_trained_opt, mean_test_score = self.optimize_KNN_params(self.X_train, self.y_train)
        KNN_score = self.score_KNN(KNN_trained_opt, self.X_test, self.y_test)
        end = time.time()
        self.param_plot_KNN(mean_test_scores=mean_test_score)
        print(f'Raw data KNN optimal score = {KNN_score}')
        print(f'Time taken = {end-start}')

    # model 
    def train_KNN(self, X, y, n_neighbors, weights):
        return KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights).fit(X, y)

    def score_KNN(self, trained_KNN_model, X_test, y_test):
        return trained_KNN_model.score(X_test, y_test)

    def predict_KNN(self, trained_KNN_model, X_test):
        return trained_KNN_model.predict(X_test)

    def optimize_KNN_params(self, X_train, y_train, n_neighbors_options=n_neighbors_options, weights_options=weights_options, cv=10, scoring='accuracy'):
        KNN_raw = KNeighborsClassifier()
        cv_train_model = GridSearchCV(KNN_raw, param_grid={'n_neighbors': n_neighbors_options, 'weights': weights_options}, cv=cv, scoring=scoring).fit(X_train, y_train)
        mean_test_score = cv_train_model.cv_results_['mean_test_score']
        print(f'Best KNN parameters: n_neighbors = {cv_train_model.best_params_["n_neighbors"]}, weights = {cv_train_model.best_params_["weights"]}')
        best_model = self.train_KNN(X_train, y_train, cv_train_model.best_params_["n_neighbors"], cv_train_model.best_params_["weights"])
        return best_model, mean_test_score.reshape(len(self.n_neighbors_options), len(self.weights_options))

    # visualization 
    def param_plot_KNN(self, mean_test_scores):
        fig = plt.figure()
        ax = sns.heatmap(mean_test_scores, xticklabels=self.weights_options, yticklabels=self.n_neighbors_options, cbar_kws={'label':'Mean Test Score'})
        plt.yticks(rotation=45)
        ax.set_yticks(ax.get_yticks()[::5])
        ax.set_ylabel('N neighbors')
        ax.set_xlabel('Weights')
        ax.set_title('CV Scores for KNN parameters')
        plt.show()
        sf.best_save(fig, 'KNN_params')

# LDA
class pLDA():
    # best parameter 
    solver_best = 'svd'
    # parameter options tested 
    solvers = ['svd', 'lsqr', 'eigen']

    def __init__(self):
        # method calls 
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data(perform_scale=False, sub_data_section='')
        # Raw data 
        start = time.time()
        LDA_trained_opt, mean_test_score = self.optimize_LDA_params(self.X_train, self.y_train)
        LDA_score = self.score_LDA(LDA_trained_opt, self.X_test, self.y_test)
        self.param_plot_LDA(mean_test_score=mean_test_score)
        end = time.time()
        print(f'Raw data LDA optimal score = {LDA_score}')
        print(f'Time taken = {end-start}')


    # model 
    def train_LDA(self, X, y, solver='svd'):
        if solver == 'svd':
            return LinearDiscriminantAnalysis(solver = solver).fit(X, y)
        else:
            return LinearDiscriminantAnalysis(solver = solver, shrinkage='auto').fit(X, y)

    def score_LDA(self, trained_LDA_model, X_test, y_test):
        return trained_LDA_model.score(X_test, y_test)

    def predict_LDA(self, trained_LDA_model, X_test):
        return trained_LDA_model.predict(X_test)
    
    def optimize_LDA_params(self, X_train, y_train, solver_options=solvers, cv=10, scoring='accuracy'):
        LDA_raw = LinearDiscriminantAnalysis()
        cv_train_model = GridSearchCV(LDA_raw, param_grid={'solver':solver_options}, cv=cv, scoring=scoring).fit(X_train, y_train)
        mean_test_score = cv_train_model.cv_results_['mean_test_score']
        print(f'Best LDA parameters: solver = {cv_train_model.best_params_["solver"]}')
        best_model = self.train_LDA(X_train, y_train, solver=cv_train_model.best_params_["solver"])
        return best_model, mean_test_score

    # visualization 
    def param_plot_LDA(self, mean_test_score):
        fig = plt.figure()
        ax = sns.barplot(x=self.solvers, y=mean_test_score)
        plt.xlabel('LDA solvers')
        plt.ylabel('CV Scores for LDA parameter')
        plt.show()
        sf.best_save(fig, 'LDA_params')

# Logistic Regression 
class pLogisticRegression():
    # best parameters
    penalty_best = 'l2'
    C_best = 0.7525
    intercept_best = True
    l1_ratio_best = 0
    # parameter options tested
    penalty_options = ['l1', 'l2', 'elasticnet', 'none']
    C_options = np.linspace(0.01, 1, 5)
    intercept_options = [True, False]
    l1_ratio_options = np.linspace(0, 1, 5)
    
    def __init__(self):
        # method calls 
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data(perform_scale=True, sub_data_section='')
        self.un_classifiers = np.unique(self.y_train)
        # Raw data 
        start = time.time()
        LR_trained_opt = self.optimize_LogisticRegression_params(self.X_train, self.y_train, self.penalty_options, self.C_options, self.intercept_options, self.l1_ratio_options)
        LR_score = self.score_LogisticRegression(LR_trained_opt, self.X_test, self.y_test)
        end = time.time()
        print(f'Raw data LogisticRegression optimal score = {LR_score}')
        print(f'Time taken = {end-start}')

    # model
    def train_LogisticRegression(self, X, y, penalty, C, fit_B0, l1_ratio):
        if penalty == 'elasticnet':
            return LogisticRegression(penalty=penalty, C=C, fit_intercept=fit_B0, l1_ratio=l1_ratio, n_jobs=4, solver='saga').fit(X, y)
        else: 
            return LogisticRegression(penalty=penalty, C=C, fit_intercept=fit_B0, n_jobs=4, solver='saga').fit(X, y)

    def score_LogisticRegression(self, trained_LogisticRegression_model, X_test, y_test):
        return trained_LogisticRegression_model.score(X_test, y_test)

    def predict_LogisticRegression(self, trained_LogisticRegression_model, X_test):
        return trained_LogisticRegression_model.predict(X_test)

    def optimize_LogisticRegression_params(self, X_train, y_train, penalty_options=penalty_options, C_options=C_options, intercept_options=intercept_options, l1_ratio_options=l1_ratio_options, cv=10, scoring='accuracy'):
        LogisticRegression_raw = LogisticRegression()
        cv_train_model = GridSearchCV(LogisticRegression_raw, param_grid={'penalty':penalty_options, 'C': C_options, 'fit_intercept':intercept_options, 'l1_ratio':l1_ratio_options}, cv=cv, scoring=scoring).fit(X_train, y_train)
        print(f'Best LogisticRegression parameters: penalty = {cv_train_model.best_params_["penalty"]}, C = {cv_train_model.best_params_["C"]}, fit_intercept = {cv_train_model.best_params_["fit_intercept"]}, l1_ratio = {cv_train_model.best_params_["l1_ratio"]}')
        best_model = self.train_LogisticRegression(X_train, y_train, penalty=cv_train_model.best_params_["penalty"], C=cv_train_model.best_params_["C"], fit_B0=cv_train_model.best_params_["fit_intercept"], l1_ratio=cv_train_model.best_params_["l1_ratio"])
        return best_model    

# QDA 
class pQDA():
    # best parameters
    reg_param_best = 0.145
    # parameters tested 
    reg_param_options = np.linspace(0, 0.3, 50)

    def __init__(self):
        # method calls 
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data(perform_scale=False, sub_data_section='')
        # Raw data 
        start = time.time()
        QDA_trained_opt, mean_test_score = self.optimize_QDA_params(self.X_train, self.y_train)
        QDA_score = self.score_QDA(QDA_trained_opt, self.X_test, self.y_test)
        self.param_plot_QDA(mean_test_score=mean_test_score)
        end = time.time()
        print(f'Raw data QDA optimal score = {QDA_score}')
        print(f'Time taken = {end-start}')
        # print(f'Predictions = {self.predict_QDA(QDA_trained_opt, self.X_test)}')

    # model 
    def train_QDA(self, X, y, reg_param):
        return QuadraticDiscriminantAnalysis(reg_param=reg_param).fit(X, y)
    
    def score_QDA(self, trained_QDA_model, X_test, y_test):
        return trained_QDA_model.score(X_test, y_test) 

    def predict_QDA(self, trained_QDA_model, X_test):
        return trained_QDA_model.predict(X_test)

    def optimize_QDA_params(self, X_train, y_train, reg_param_options=reg_param_options, cv=10, scoring='accuracy'):
        QDA_raw = QuadraticDiscriminantAnalysis()
        cv_train_model = GridSearchCV(QDA_raw, param_grid={'reg_param': reg_param_options}, cv=cv, scoring=scoring).fit(X_train, y_train)
        mean_test_score = cv_train_model.cv_results_['mean_test_score']
        print(f'Best QDA parameters: reg_param = {cv_train_model.best_params_["reg_param"]}')
        best_model = self.train_QDA(X_train, y_train, reg_param=cv_train_model.best_params_["reg_param"])
        return best_model, mean_test_score

    # visualization 
    def param_plot_QDA(self, mean_test_score):
        fig = plt.figure()
        ax = sns.barplot(x=np.round(self.reg_param_options,2), y=mean_test_score)
        ax.set_xticks(ax.get_xticks()[::5])
        plt.xticks(rotation=45)
        plt.xlabel('Regulating parameter')
        plt.ylabel('CV Scores for QDA parameter')
        plt.show()
        sf.best_save(fig, 'QDA_params')

# SVM
class pSVM():
    # best parameters 
    C_param_best = 8.68 
    kernel_best = 'linear' 
    # parameters tesed
    C_param_guesses = np.linspace(3, 30, 20)
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']

    def __init__(self):
        # method calls 
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data(perform_scale=True, sub_data_section='')
        # Raw data 
        start = time.time()
        SVM_trained_opt, mean_test_score = self.optimize_SVM_params(self.X_train, self.y_train, self.C_param_guesses, self.kernels)
        SVM_score = self.score_SVM(SVM_trained_opt,self.X_test, self.y_test)
        self.param_plot_SVM(mean_test_scores=mean_test_score)
        end = time.time()
        print(f'Raw data SVM optimal score = {SVM_score}') 
        print(f'Time taken = {end-start}')

    # model 
    def train_SVM(self, X, y, C=1, kernel='rbf', maxIter=1.6e5):
        return SVC(kernel=kernel, C=C, max_iter=maxIter).fit(X, y)

    def score_SVM(self, trained_SVM_model, X_test, y_test):
        return trained_SVM_model.score(X_test, y_test)

    def predict_SVM(self, trained_SVM_model, X_test):
        return trained_SVM_model.predict(X_test)

    # parameter estimation 
    def optimize_SVM_params(self, X_train, y_train, C_options=C_param_guesses, kernel_options=kernels, cv=10, scoring='accuracy'):
        SVM_raw = SVC()
        cv_train_model = GridSearchCV(SVM_raw, param_grid={'C': C_options, 'kernel':kernel_options}, cv=cv, scoring=scoring).fit(X_train, y_train)
        mean_test_score = cv_train_model.cv_results_['mean_test_score']
        print(f'Best SVM parameters: C = {cv_train_model.best_params_["C"]}, kernel = {cv_train_model.best_params_["kernel"]}')
        best_model = self.train_SVM(X_train, y_train, C=cv_train_model.best_params_["C"], kernel=cv_train_model.best_params_["kernel"])
        return best_model, mean_test_score.reshape(len(self.C_param_guesses), len(self.kernels))

    # visualization
    def param_plot_SVM(self, mean_test_scores):
        fig = plt.figure()
        ax = sns.heatmap(mean_test_scores, xticklabels=self.kernels, yticklabels=self.C_param_guesses, cbar_kws={'label':'Mean Test Score'})
        ax.set_xlabel('Kernels')
        ax.set_ylabel('C')
        ax.set_title('CV Scores for SVM parameters')
        plt.show()
        sf.best_save(fig, 'SVM_params')
        