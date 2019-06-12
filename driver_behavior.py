# -*- coding: utf-8 -*-
import time
import pandas as pd

from sklearn import preprocessing, tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from pprint import pprint

class DriverBehavior:
    def __init__(self, data_path='data.csv', labels_index=52):
        self.data_df = pd.read_csv(data_path)
        
        # replaces the 'A' - 'J' class labels in column <labels_index=52> to '0' - '9'
        # note:it's a dict-like `to_replace` 
        replace_y_numbers = {k:v for v,k in enumerate(sorted(set(self.data_df.iloc[:, labels_index])))}
        self.data_df.iloc[:, labels_index] = self.data_df.iloc[:, labels_index].replace(replace_y_numbers)
        
        # the labels column name is 'Class'
        self.X = self.data_df.drop(columns=['Class', 'Time(s)', 'PathOrder'])
        self.y = self.data_df.iloc[:, labels_index]
        self.models = {
            'decision_tree' : tree.DecisionTreeClassifier(),
            'random_forest' : RandomForestClassifier(),
            'knn'           : KNeighborsClassifier(),
            'mlp'           : MLPClassifier(),
            'gradient_boosting' : GradientBoostingClassifier(),
            'linear_svc'    : LinearSVC(),
            'logistic'      : LogisticRegression(),
            'adaboost'      : AdaBoostClassifier(),
            'naive_bayes'   : GaussianNB(),
                      }
        
        self.__default_models = self.models        
        self.__current_model = ''        
        self.__fitted = {k:False for k in self.models.keys()} # set  false for all model
        
        self.standarize()
        self.__train_test_split()
        
        self.__train_accuracies = {}
        self.__test_accuracies = {}
        
        #note: here we use self.standarize() not self.normalize()
        #self.normalize()
        
        
    def normalize(self):
        min_max_scaler = preprocessing.MinMaxScaler()
        self.normalized_X = min_max_scaler.fit_transform(self.X)
        
    def standarize(self):
        scaler = preprocessing.StandardScaler()
        self.standarized_X = scaler.fit_transform(self.X) # from DataFrame to get a numpy array
        
    def __train_test_split(self, random_state = 0 ):
        # seed is set to 420 by default
        X_train, X_test, y_train, y_test = train_test_split(self.standarized_X, self.y, test_size = 0.25,
                                                                random_state=random_state)
            
        self.train_data = {'X':X_train, 'y': y_train}
        self.test_data = {'X':X_test, 'y': y_test}
    
    def train(self, model_name='decision_tree', selected_features=None):
        if model_name in self.models:
            self.__current_model = model_name
            
            if self.__fitted[model_name]:
                # reset model parameters to default,not fitted params
                self.models[model_name] = self.__default_models[model_name]
            else:            
                self.__fitted[model_name] = True
            
            X_train = self.train_data['X']
            '''if specified features,use them,else use all features ''' 
            if selected_features:
                X_train = X_train[:, selected_features]
            
            y_train = self.train_data['y']
            
            self.models[model_name].fit(X_train, y_train)            
            
    def train_accuracy(self, force_update=False, selected_features=None, select_model=None):
        predictions = {}
        accuracies = {}
        
        X_train = self.train_data['X']
        if selected_features:
            X_train = X_train[:, selected_features]
        
        if force_update or not self.__current_model:
            for name, classifier in self.models.items():
                if self.__fitted[name]:
                    predictions[name] = classifier.predict(X_train)
            for name, pred in predictions.items():
                accuracies[name] = accuracy_score(self.train_data['y'], pred, normalize=True)
            self.__train_accuracies = accuracies
            
        elif select_model is not None and select_model in self.models.keys():
            predictions = self.models[select_model].predict(X_train)
            accuracies = accuracy_score(self.train_data['y'], predictions, normalize=True)
            self.__train_accuracies[select_model] = accuracies
            
        else: # self.__current_model =True
            predictions = self.models[self.__current_model].predict(X_train)
            accuracies = accuracy_score(self.train_data['y'], predictions, normalize=True)
            self.__train_accuracies[self.__current_model] = accuracies
        
        return self.__train_accuracies
    
    def test_accuracy(self, force_update=False, selected_features=None, select_model=None):
        predictions = {}
        accuracies = {}
        
        X_test = self.test_data['X']
        if selected_features:
            X_test = X_test[:, selected_features]
        
        if force_update or not self.__current_model:
            for name, classifier in self.models.items():
                if self.__fitted[name]:
                    predictions[name] = classifier.predict(X_test)

            for name, pred in predictions.items():
                accuracies[name] = accuracy_score(self.test_data['y'], pred, normalize=True)
            self.__test_accuracies = accuracies
        
        elif select_model is not None and select_model in self.models.keys():
            predictions = self.models[select_model].predict(X_test)
            accuracies = accuracy_score(self.test_data['y'], predictions, normalize=True)
            self.__test_accuracies[select_model] = accuracies
        
        else: # self.__current_model =True
            predictions = self.models[self.__current_model].predict(X_test)
            accuracies = accuracy_score(self.test_data['y'], predictions, normalize=True)
            self.__test_accuracies[self.__current_model] = accuracies
        
        return self.__test_accuracies
    
    def print_accuracies(self, select_model=None, force_update=False):
        print('{} train accuracy:'.format(time.strftime('%Y-%m-%d %X')))
        pprint(self.train_accuracy(select_model=select_model, force_update=force_update))
        print('{} test accuracy:'.format(time.strftime('%Y-%m-%d %X')))
        pprint(self.test_accuracy(select_model=select_model, force_update=force_update))
