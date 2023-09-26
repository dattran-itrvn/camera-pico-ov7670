from keras.models import model_from_json
import cv2
import numpy as np
import os

path_model = '/home/ttb/PycharmProjects/OV7670_cam/model'
list_path_data_test = ['/home/ttb/data_test/face', '/home/ttb/data_test/no_face']
detect_face = [0, 0]
detect_no_face = [0, 0]

def load_model():
    # load json and create model
    json_file = open(path_model + '/face.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path_model + "/face.h5")
    loaded_model.summary()
    print("Loaded model from disk")

    return loaded_model

def predict(model, list_path_data):
    for path in list_path_data:
        for img_path in os.listdir(path):
            img_test_path = path + '/' + img_path
            img_data = cv2.imread(img_test_path)
            img_data_rz = cv2.resize(img_data, (60, 60))
            img_data_cvt = cv2.cvtColor(img_data_rz, cv2.COLOR_BGR2GRAY)
            img_arr_data = np.array(img_data_cvt)
            img_arr_data = img_arr_data.astype('float32')
            img_arr_data = np.expand_dims(img_arr_data, axis=2)
            img_arr_data = np.expand_dims(img_arr_data, axis=0)

            y_pred = model.predict(img_arr_data)
            if path.split('/')[-1] == "face":
                if y_pred[0][0] > y_pred[0][1]:
                    detect_face[0] += 1
                else:
                    detect_face[1] += 1
            elif path.split('/')[-1] == "no_face":
                if y_pred[0][0] < y_pred[0][1]:
                    detect_no_face[1] += 1
                else:
                    detect_no_face[0] += 1

    # print cfm
    print()
    print(f'===============================')
    print(f'||          Face     No face  ||')
    print(f'|| Face       {detect_face[0]}      {detect_face[1]}      ||')
    print(f'|| No face    {detect_no_face[0]}       {detect_no_face[1]}    ||')
    print(f'===============================')

def main():
    model = load_model()
    predict(model, list_path_data_test)

if __name__ == "__main__":
    main()