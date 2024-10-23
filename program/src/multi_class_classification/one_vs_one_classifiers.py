
import numpy as np

from sklearn.base import BaseEstimator
# from sklearn.model_selection import GridSearchCV

# Define una clase personalizada que hereda de BaseEstimator
class CustomOneVsOneClassifier(BaseEstimator):
    def __init__(self, estimator, unique_classes, weight_classes=None):
        self.estimator = estimator
        self.classifiers = []
        self.index_classifiers_list = []
        self.unique_classes = unique_classes
        self.weight_classes = weight_classes
        self.grid_params = None


    def fit(self, X, y, params, class_internal_weights=None):
        cc = 0
        for i in range(len(self.unique_classes)):
            for j in range(i + 1, len(self.unique_classes)):
                class_indices = np.where((y == self.unique_classes[i]) | (y == self.unique_classes[j]))
                X_class = X[class_indices]
                y_class = y[class_indices]
                y_new = y_class.copy()
                y_new[y_class==i] = 0
                y_new[y_class==j] = 1

                # creating new classifier with best grid seach parameters
                if class_internal_weights is not None:
                    params["class_weight"]=class_internal_weights[cc]
                    cc +=1
                _classifier = self.estimator(**params)
                _classifier.fit(X_class, y_new)
                self.classifiers.append(_classifier)
                self.index_classifiers_list.append((i,j))


    # def fit_grid_search(self, X, y, grid_params, metric="accuracy", cv=5):
    #     self.grid_params = grid_params

    #     for i in range(len(self.unique_classes)):
    #         for j in range(i + 1, len(self.unique_classes)):
    #             class_indices = np.where((y == self.unique_classes[i]) | (y == self.unique_classes[j]))
    #             X_class = X[class_indices]
    #             y_class = y[class_indices]
    #             y_new = y_class.copy()
    #             y_new[y_class==i] = 0
    #             y_new[y_class==j] = 1
    
    #             # Crear un clasificador específico para el par de clases actual
    #             classifier = self.estimator
    #             grid_search = GridSearchCV(classifier(), self.grid_params, cv=cv, n_jobs=-1, scoring=metric)
    #             grid_search.fit(X_class, y_new)

    #             # creating new classifier with best grid seach parameters
    #             _classifier = classifier(**grid_search.best_params_)
    #             _classifier.fit(X_class, y_new)
    #             self.classifiers.append(_classifier)
    #             self.index_classifiers_list.append((i,j))
    #             # print(self.index_classifiers_list[-1])
    #             print(grid_search.best_params_)
    #             # print(grid_search.best_score_)


    def compute_sum_probs(self, X):
        sum_probs = np.zeros((X.shape[0], len(self.unique_classes)))
        for i, classifier in enumerate(self.classifiers):
            index_0, index_1 = self.index_classifiers_list[i]

            classifier_probs = classifier.predict_proba(X)
            sum_probs[:,index_0] = sum_probs[:,index_0] + classifier_probs[:, 0]
            sum_probs[:,index_1] = sum_probs[:,index_1] + classifier_probs[:, 1]

        if self.weight_classes is not None:
            sum_probs *= np.array(self.weight_classes)[np.newaxis,:]
            # for i, w in enumerate(self.weight_classes):
                # if w != 1:
                # sum_probs[:,i] = sum_probs[:,i] * w
        return sum_probs


    def predict(self, X):
        sum_probs = self.compute_sum_probs(X)
        sum_rows = np.sum(sum_probs, axis=1)
        normalized_probs = sum_probs/sum_rows[:, np.newaxis]
        return np.array([self.unique_classes[np.argmax(prob)] for prob in sum_probs]), normalized_probs


    def predict_proba(self, X):
        sum_probs = self.compute_sum_probs(X)
        sum_rows = np.sum(sum_probs, axis=1)
        normalized_probs = sum_probs/sum_rows[:, np.newaxis]
        return normalized_probs


