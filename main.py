
import cv2
from program.multi_class_melanoma_detector import MultiClassMelanomaDetector
from program.model_creator_from_features import ModelCreatorFromFeatures
import security_check
 

def create_models():
    bin_mel_classification = ModelCreatorFromFeatures()
    print("Train -------------------------------------------------")
    # bin_mel_classification.train_from_features(use_val=False) # model_v0_only_train.pkl
    bin_mel_classification.train_from_features(use_val=True) # model_v0_train_val.pkl
    # print("Val -------------------------------------------------")
    # bin_mel_classification.val_from_features()


def test_program():
    # img_path_name = "C:/Users/Agustin/portfolio/medical_imaging_projects/s3_skin_lesion_detection_ml/code/multiclass/program/images/original/train/mel/mel00012.jpg"
    # img_path_name = "C:/Users/Agustin/portfolio/medical_imaging_projects/s3_skin_lesion_detection_ml/code/multiclass/program/images/original/train/bcc/bcc00007.jpg"
    img_path_name = "C:/Users/Agustin/portfolio/medical_imaging_projects/s3_skin_lesion_detection_ml/code/multiclass/program/images/original/train/scc/scc00010.jpg"
    img = cv2.imread(img_path_name)

    security_message, is_secure = security_check.check_image(img)
    if not is_secure:
        raise Exception('Security check failed: {}'.format(security_message))
     
    pred, prob, img_result = MultiClassMelanomaDetector().test(img)
    print(f"Prediction: {pred}, Probability: {prob}")

    cv2.imshow("img_result", img_result)
    cv2.waitKey(0)


if __name__ == '__main__':
    # create_models()
    test_program()
    
