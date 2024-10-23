
import pandas as pd
import pathlib
import os
import numpy as np

from sklearn.metrics import classification_report, cohen_kappa_score

from .src.multi_class_classification.classifier import Classifier

def get_base_folder():
    return str(pathlib.Path(__file__).parent.resolve()).replace(os.sep, "/") + "/"

    
class ModelCreatorFromFeatures:

    F_FOLDER_POSTPROCESSING_V0 = "postprocessing_v0"
    F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1 = "postprocessing_v0_mask_combined_v1"
    F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1_I = "postprocessing_v0_mask_combined_v1_i"
    F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1_BORDER = "postprocessing_v0_mask_combined_v1_border"

    F_COLOR = "color"
    F_TEXTURE_LBP = "texture_lbp"
    F_TEXTURE_GLCM = "texture_glcm"
    F_SHAPE = "shape"

    F_COLOR_SPATIAL = "color_spatial"
    F_TEXTURE_GLCM_SPATIAL = "texture_glcm_spatial"
    F_TEXTURE_LBP_SPATIAL = "texture_lbp_spatial"

    MAX_INDEX_TRAIN_MEL = 2713
    MAX_INDEX_TRAIN_BCC = 1993
    MAX_INDEX_TRAIN_SCC = 376

    # MODEL_PATH_NAME = "model_v0_only_train.pkl"
    MODEL_PATH_NAME = "model_v0.pkl"


    def train_from_features(self, use_val=False):
        # load train features and labels
        def load_train_features(folder, features_name):
            file_np = pd.read_csv(f"{get_base_folder()}features/train_val/{folder}/{features_name}.csv").values
            if use_val:
                features = file_np[:, 2:]
                return features
            else:
                data_mel = file_np[file_np[:, 1] == 0]
                data_bcc = file_np[file_np[:, 1] == 1]
                data_scc = file_np[file_np[:, 1] == 2]
                features =  np.concatenate((data_mel[:self.MAX_INDEX_TRAIN_MEL,2:], data_bcc[:self.MAX_INDEX_TRAIN_BCC,2:], data_scc[:self.MAX_INDEX_TRAIN_SCC,2:]), axis=0)
                # features =  np.concatenate((data_mel[:500,2:], data_bcc[:500,2:], data_scc[:100,2:]), axis=0)
                return features
        
        def load_train_labels():
            file_np = pd.read_csv(f"{get_base_folder()}features/train_val/{self.F_FOLDER_POSTPROCESSING_V0}/{self.F_COLOR}.csv").values
            if use_val:
                labels = file_np[:, 1]
                return labels
            else:
                data_mel = file_np[file_np[:, 1] == 0]
                data_bcc = file_np[file_np[:, 1] == 1]
                data_scc = file_np[file_np[:, 1] == 2]
                labels =  np.concatenate((data_mel[:self.MAX_INDEX_TRAIN_MEL,1], data_bcc[:self.MAX_INDEX_TRAIN_BCC,1], data_scc[:self.MAX_INDEX_TRAIN_SCC,1]), axis=0)
                # labels =  np.concatenate((data_mel[:500,1], data_bcc[:500,1], data_scc[:100,1]), axis=0)
                return labels


        labels = load_train_labels()
        train_features_dict = {
            "image": {
                "color": load_train_features(self.F_FOLDER_POSTPROCESSING_V0, self.F_COLOR),
                "texture_lbp": load_train_features(self.F_FOLDER_POSTPROCESSING_V0, self.F_TEXTURE_LBP),
                "texture_glcm": load_train_features(self.F_FOLDER_POSTPROCESSING_V0, self.F_TEXTURE_GLCM),
            },
            "mask": {
                "color": load_train_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1, self.F_COLOR),
                "texture_lbp": load_train_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1, self.F_TEXTURE_LBP),
                "texture_glcm": load_train_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1, self.F_TEXTURE_GLCM),
                "shape": load_train_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1, self.F_SHAPE),
            },
            "mask_inv": {
                "color": load_train_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1_I, self.F_COLOR),
                "texture_lbp": load_train_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1_I, self.F_TEXTURE_LBP),
                "texture_glcm": load_train_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1_I, self.F_TEXTURE_GLCM),
            },
            "mask_border": {
                "color": load_train_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1_BORDER, self.F_COLOR),
                "texture_lbp": load_train_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1_BORDER, self.F_TEXTURE_LBP),
                "texture_glcm": load_train_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1_BORDER, self.F_TEXTURE_GLCM),
            },
            "spatial": {
                "color": load_train_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1, self.F_COLOR_SPATIAL),
                "texture_lbp": load_train_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1, self.F_TEXTURE_LBP_SPATIAL),
                "texture_glcm": load_train_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1, self.F_TEXTURE_GLCM_SPATIAL),
            },
        }

        # train
        classifier = Classifier()
        classifier.train(train_features_dict, labels)

        # save
        classifier.save(f"{get_base_folder()}models/{self.MODEL_PATH_NAME}")

        # test predictions
        # labels_pred, individual_pred = classifier.test(train_features_dict)

        # report = classification_report(labels, labels_pred)
        # kappa = cohen_kappa_score(labels, labels_pred)
        # print("Reporte de clasificación:\n", report)
        # print(f'Kappa: {kappa}')

        # for i in range(len(individual_pred)):
        #     kappa = cohen_kappa_score(labels, individual_pred[i])
        #     print(f"Kappa C{i+1}: {kappa}")


    def val_from_features(self):
        # load val features and labels
        def load_val_features(folder, features_name):
            file_np = pd.read_csv(f"{get_base_folder()}features/train_val/{folder}/{features_name}.csv").values
            data_mel = file_np[file_np[:, 1] == 0]
            data_bcc = file_np[file_np[:, 1] == 1]
            data_scc = file_np[file_np[:, 1] == 2]
            features = np.concatenate((data_mel[self.MAX_INDEX_TRAIN_MEL:,2:], data_bcc[self.MAX_INDEX_TRAIN_BCC:,2:], data_scc[self.MAX_INDEX_TRAIN_SCC:,2:]), axis=0)
            return features
            
        def load_val_labels():
            file_np = pd.read_csv(f"{get_base_folder()}features/train_val/{self.F_FOLDER_POSTPROCESSING_V0}/{self.F_COLOR}.csv").values
            data_mel = file_np[file_np[:, 1] == 0]
            data_bcc = file_np[file_np[:, 1] == 1]
            data_scc = file_np[file_np[:, 1] == 2]
            labels = np.concatenate((data_mel[self.MAX_INDEX_TRAIN_MEL:,1], data_bcc[self.MAX_INDEX_TRAIN_BCC:,1], data_scc[self.MAX_INDEX_TRAIN_SCC:,1]), axis=0)
            return labels

        labels = load_val_labels()
        val_features_dict = {
            "image": {
                "color": load_val_features(self.F_FOLDER_POSTPROCESSING_V0, self.F_COLOR),
                "texture_lbp": load_val_features(self.F_FOLDER_POSTPROCESSING_V0, self.F_TEXTURE_LBP),
                "texture_glcm": load_val_features(self.F_FOLDER_POSTPROCESSING_V0, self.F_TEXTURE_GLCM),
            },
            "mask": {
                "color": load_val_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1, self.F_COLOR),
                "texture_lbp": load_val_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1, self.F_TEXTURE_LBP),
                "texture_glcm": load_val_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1, self.F_TEXTURE_GLCM),
                "shape": load_val_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1, self.F_SHAPE),
            },
            "mask_inv": {
                "color": load_val_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1_I, self.F_COLOR),
                "texture_lbp": load_val_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1_I, self.F_TEXTURE_LBP),
                "texture_glcm": load_val_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1_I, self.F_TEXTURE_GLCM),
            },
            "mask_border": {
                "color": load_val_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1_BORDER, self.F_COLOR),
                "texture_lbp": load_val_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1_BORDER, self.F_TEXTURE_LBP),
                "texture_glcm": load_val_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1_BORDER, self.F_TEXTURE_GLCM),
            },
            "spatial": {
                "color": load_val_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1, self.F_COLOR_SPATIAL),
                "texture_lbp": load_val_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1, self.F_TEXTURE_LBP_SPATIAL),
                "texture_glcm": load_val_features(self.F_FOLDER_POSTPROCESSING_V0_MASK_COMBINED_V1, self.F_TEXTURE_GLCM_SPATIAL),
            },
        }

        # classification
        classifier = Classifier()
        classifier.load(f"{get_base_folder()}models/{self.MODEL_PATH_NAME}")
        labels_pred, individual_pred = classifier.test(val_features_dict)

        # results
        report = classification_report(labels, labels_pred)
        kappa = cohen_kappa_score(labels, labels_pred)
        print("Reporte de clasificación:\n", report)
        print(f'Kappa: {kappa}')

        for i in range(len(individual_pred)):
            kappa = cohen_kappa_score(labels, individual_pred[i])
            print(f"Kappa C{i+1}: {kappa}")




# Reporte de clasificación:
#                precision    recall  f1-score   support

#          0.0       0.92      0.92      0.92       678
#          1.0       0.86      0.88      0.87       498
#          2.0       0.62      0.57      0.60        94

#     accuracy                           0.88      1270
#    macro avg       0.80      0.79      0.80      1270
# weighted avg       0.88      0.88      0.88      1270

# Kappa: 0.7784556754798337
# Kappa C1: 0.7678058612632217
# Kappa C2: 0.7632723311352302