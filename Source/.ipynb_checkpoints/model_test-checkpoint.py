import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, learning_curve
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class Test:
    def __init__(self, model, n_folds = 5):
        self.n_folds = n_folds
        self.model = model
        
    def get_model(self):
        return self.model
        
        
    def cross_validation_scores(self, X_train, y_train):
        dd = {}
        model = self.model
        f1 = []
        acc = []
        rec = []
        prec = []        

        if type(model).__name__ in ['LogisticRegression', 'RandomForestClassifier', 'GaussianNB']:   
            f1 = cross_val_score(model, X_train, y_train, cv = StratifiedKFold(n_splits = self.n_folds, shuffle = True, random_state = 0),                                      scoring = 'f1')
            acc = cross_val_score(model, X_train, y_train, cv = StratifiedKFold(n_splits = self.n_folds, shuffle = True, random_state = 0),                                     scoring = 'accuracy')
            rec = cross_val_score(model, X_train, y_train, cv = StratifiedKFold(n_splits = self.n_folds, shuffle = True, random_state = 0),                                     scoring = 'recall')
            prec = cross_val_score(model, X_train, y_train, cv = StratifiedKFold(n_splits = self.n_folds, shuffle = True, random_state = 0),                                     scoring = 'precision')
            
            
        elif type(model).__name__ == 'FCM':
            kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
            for train, test in kfold.split(X_train.values, y_train):
                model.fit(X_train.values[train])
                predictions = model.predict(X_train.values[test])
                acc.append(accuracy_score(y_train[test], predictions))
                prec.append(precision_score(y_train[test], predictions))
                rec.append(recall_score(y_train[test], predictions))
                f1.append(f1_score(y_train[test], predictions))
                
        elif type(model).__name__ in ['GaussianMixture', 'KMeans']:
            kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
            for train, test in kfold.split(X_train, y_train):
                model.fit(X_train.iloc[train])
                predictions = self.model.predict(X_train.iloc[test])
                acc.append(accuracy_score(y_train.iloc[test], predictions))
                prec.append(precision_score(y_train.iloc[test], predictions))
                rec.append(recall_score(y_train.iloc[test], predictions))
                f1.append(f1_score(y_train.iloc[test], predictions))
        
        dd = {'Mean': [np.array(acc).mean(), np.array(rec).mean(), np.array(prec).mean(), np.array(f1).mean()],
            'STD': [np.array(acc).std(), np.array(rec).std(), np.array(prec).std(), np.array(f1).std()]}
            
            
        
        return pd.DataFrame(dd, index = ['Accuracy', 'Recall', 'Precision', 'F1_Score'])
    
    
    def train_test_report(self, X_train, y_train, X_test, y_test, supervised = True):
        model = self.model        
        if supervised == True:
            model.fit(X_train, y_train)
        else: 
            model.fit(X_train)
        predictions = self.model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions)
        rec = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        print('Accuracy: {}, Recall: {}, Precision: {}, F1_Score: {}'.format(acc, rec, prec, f1))
        print(classification_report(y_test, predictions))
        plt.figure()
        sns.heatmap(confusion_matrix(y_test, predictions), annot = True, fmt='g', cmap='Blues')
        
        if supervised == True:
            N, train_score, val_score = learning_curve(model, X_train, y_train, cv = 5, train_sizes = np.linspace(0.1, 1, 10),
                                                       scoring = 'f1')
            plt.figure()
            plt.plot(N, train_score.mean(axis = 1), label = 'Train score')
            plt.plot(N, val_score.mean(axis = 1), label = 'Validation score')
            plt.title('Learning curve')
            plt.legend()
            plt.show()
        return model
   

    def predict_on_test(self, X_test, y_test):
        dd = {}
        model = self.model 
        dd = {'True': y_test, 'Predictions': model.predict(X_test), 'Proba of Dysgraphia': model.predict_proba(X_test)[:, 1]*100}
        df = pd.DataFrame(dd)
        return df
    
    
    def precision_recall_curve(self, X_train, y_train):
        model = self.model
        precision = recall = threshold = []
        if type(model).__name__ == 'LogisticRegression':
            precision, recall, threshold = precision_recall_curve(y_train, model.decision_function(X_train))
        elif type(model).__name__ == 'RandomForestClassifier':
            precision, recall, threshold = precision_recall_curve(y_train, model.predict_proba(X_train)[:, 1])
        plt.figure()
        plt.plot(threshold, precision[:-1], label = 'Precision')
        plt.plot(threshold, recall[:-1], label = 'Recall')
        plt.title('Precision Recall curve')
        plt.legend()
        
    def test_on_synthetic_data(self, syn_features, syn_targets):
        model = self.model
        predictions = model.predict(syn_features)
        acc = accuracy_score(syn_targets, predictions)
        prec = precision_score(syn_targets, predictions)
        rec = recall_score(syn_targets, predictions)
        f1 = f1_score(syn_targets, predictions)
        print('Accuracy: {}, Recall: {}, Precision: {}, F1_Score: {}'.format(acc, rec, prec, f1))

        