
import numpy as np
import joblib
import threading
from sklearn.svm import SVC

from .features_engeneer import FeaturesEngeneer
from .one_vs_one_classifiers import CustomOneVsOneClassifier

from .classifier_combination import combiner


class Classifier:

    PARAMS = {'C': 10, 'degree': 5, 'gamma': 'scale', 'kernel': 'rbf', 'probability': True, "random_state":42}
    WEIGHT_CLASSES = [1,1,1.42]

    # CLASSIFIER_WEIGHTS = np.array([0.8587987355110642,0.8379873551106428,0.8163856691253951,0.7987355110642782,0.821654373024236])


    def __init__(self):
        self.features_engeneer = FeaturesEngeneer()
        self.c1 = CustomOneVsOneClassifier(SVC, [0,1,2], self.WEIGHT_CLASSES)
        self.c2 = CustomOneVsOneClassifier(SVC, [0,1,2], self.WEIGHT_CLASSES)
        self.trained = False
        

    # Multithreaded training
    def train(self, features_dict, labels):

        # feature engineering and get features for classifiers  
        features_dict_pca = self.features_engeneer.fit_transform(features_dict)
        x_c1, x_c2 = self.__get_features_for_classifier(features_dict_pca)

        print("fit C1")
        self.c1.fit(x_c1, labels, self.PARAMS)
        print("fit C2")
        self.c2.fit(x_c2, labels, self.PARAMS)

        self.trained = True


    def test(self, features_dict, return_proba=False):
        if not self.trained:
            raise Exception('Classificator not trained')
        
        # feature engineering and get features for classifiers
        features_dict_pca = self.features_engeneer.transform(features_dict)
        x_c1, x_c2 = self.__get_features_for_classifier(features_dict_pca)
        
        return self.__predict_combined(x_c1, x_c2, return_proba=return_proba)


    def __predict_combined(self, x_c1, x_c2, return_proba=False):
        
        scores = np.zeros((2, x_c1.shape[0], 3))
        c_predictions = np.zeros((2, x_c1.shape[0]))
        
        c_predictions[0], scores[0] = self.c1.predict(x_c1)
        c_predictions[1], scores[1] = self.c2.predict(x_c2)

        combined_scores = combiner.combine_results_multiclass(scores)
        # ---- compute results
        y_combined = np.array([np.argmax(prob) for prob in combined_scores])
        
        if return_proba:
            return y_combined, c_predictions, combined_scores, scores
        else:
            return y_combined, c_predictions
    

    def __get_features_for_classifier(self, features_dict_pca):
        # features for each classifier
        x_to_concat_c1 = []
        x_to_concat_c2 = []
        for feature_type, f_dict in features_dict_pca.items():
            for feature_name, array in f_dict.items():
                if feature_type != "spatial":
                    x_to_concat_c1.append(array)
                x_to_concat_c2.append(array)


        x_c1 = np.hstack(x_to_concat_c1)
        x_c2 = np.hstack(x_to_concat_c2)

        return x_c1, x_c2
        

    def save(self, filepath):
        model_data = {
            'features_engeneer': self.features_engeneer,
            'c1': self.c1,
            'c2': self.c2,
        }
        joblib.dump(model_data, filepath)
        print(f'Model saved in {filepath}')


    def load(self, filepath):
        model_data = joblib.load(filepath)
        self.features_engeneer = model_data['features_engeneer']
        self.c1 = model_data['c1']
        self.c2 = model_data['c2']
        self.trained = True
        print(f'Model loaded from {filepath}')
