from __future__ import division
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import recall_score, precision_score, f1_score, \
                            accuracy_score, confusion_matrix, roc_curve, auc
from itertools import izip
from sklearn.model_selection import GridSearchCV, cross_val_score
from scipy.stats import hmean
from Plotting import plot_clf_scores
import cPickle


class Classifiers(object):
    '''
    Classifier object for fitting, storing, and comparing multiple model output
    '''
    def __init__(self, classifier_list):
        self.classifiers = classifier_list
        self.classifier_names = [est.__class__.__name__ for est in self.classifiers]

    def train(self, X, y):
        # self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(X, y,
        #                test_size=0.25, random_state=42)

        for clf, name in izip(self.classifiers,self.classifier_names):
            # clf.fit(self._X_train, self._y_train)
            clf.fit(X, y)

            with open('Fitted_Model_{}_Style'.format(name), 'wb') as f:
                cPickle.dump(clf, f)

    def cross_validate(self, X, y):
        params_list = [{'n_estimators': [60, 70, 80, 90, 100],
                        'learning_rate': [0.01, 0.05, 0.1]},
                       {'n_estimators': [50, 60, 70, 90, 100, 110, 120],
                        'learning_rate': [0.05, 0.5, 1],
                        'max_depth':[1, 3, 5, 10]},
                       {'n_estimators': [100, 500], 'max_depth':[20, 50],
                        'criterion':['gini']}]

        for clf, params in zip(self.classifiers[2], params_list[2]):
            print("\n____________{}____________".format(clf.__class__.__name__))

            gscv = GridSearchCV(clf, params)
            clf = gscv.fit(self._X_train, self._y_train)
            print 'Best parameters: %s' % clf.best_params_
            print 'Best F1 score: %s' % clf.best_score_

            scores = cross_val_score(clf, self._X_train, self._y_train, cv=5,
                                     scoring='accuracy')
            print "Accuracy: {:.3%}".format(np.mean(scores))

    def plot_roc_curve(self):
        fig, ax = plt.subplots()
        for name, clf in zip(self.classifier_names, self.classifiers):
            predict_probas = clf.predict_proba(self._X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(self._y_test, predict_probas,
                                             pos_label=1)
            roc_auc = auc(x=fpr, y=tpr)
            ax.plot(fpr, tpr, label='{} (AUC = {:.2f})'.format(name, roc_auc))

        # 45 degree line
        x_diag = np.linspace(0, 1.0, 20)
        ax.plot(x_diag, x_diag, color='grey', ls='--')
        ax.legend(loc='best')
        ax.set_ylabel('True Positive Rate', size=20)
        ax.set_xlabel('False Positive Rate', size=20)
        ax.tick_params(axis='both', which='major', labelsize=16)
        fig.set_size_inches(15, 10)
        fig.savefig('ROC_curves_no_newssites.png', dpi=100)

    def f1_harmonic_mean(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)

        TN = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]
        TP = cm[1][1]

        if TN == 0:
            TN = 1
        if TP == 0:
            TP = 1

        inv_precision = TN/(TN+FN)  # Proportion of those identified as negative that actually are.
        inv_recall = TN/(FP+TN)  # Proportion of those *actually*  negative identified as such.

        inverted_f1 = hmean([inv_recall, inv_precision])
        harmonic_f1 = hmean([f1_score(y_true, y_pred), inverted_f1])
        mean_f1 = np.mean([f1_score(y_true, y_pred), inverted_f1])

        return harmonic_f1, mean_f1

    def test(self):
        f1_means = []
        for name, clf in zip(self.classifier_names, self.classifiers):
            print '{} results:'.format(name)
            predictions = clf.predict(self._X_test)

            # print "F1: {:.3%}".format(f1_score(self._y_test, predictions))
            harmonic, regular = self.f1_harmonic_mean(self._y_test, predictions)
            # print "Harmonic f1: {:.3%}".format(harmonic)

            f1_means.append(regular)

            print "Mean f1: {:.3%}".format(regular)
            print "\n"
        plot_clf_scores(f1_means, self.classifier_names)
