import numpy as np
import cv2
import os
from tensorflow.lite.python.interpreter import Interpreter

path_tflite = '/home/ttb/PycharmProjects/OV7670_cam/model/int_quant_model.tflite'
list_path_data_test = ['/home/ttb/data_test/face', '/home/ttb/data_test/no_face']
detect_face = [0, 0]
detect_no_face = [0, 0]

def load_model(path_model):
  interpreter = Interpreter(model_path=path_model)
  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  input_type = interpreter.get_input_details()[0]['dtype']
  print('input: ', input_type)
  output_type = interpreter.get_output_details()[0]['dtype']
  print('output: ', output_type)

  return interpreter, input_details, output_details


def predict(interpreter, input_details, output_details, list_path_data):
  test_image = None
  for path in list_path_data:
    for img_path in os.listdir(path):
      img_test_path = path + '/' + img_path
      img_data = cv2.imread(img_test_path)
      img_data_rz = cv2.resize(img_data, (60, 60))
      img_data_cvt = cv2.cvtColor(img_data_rz, cv2.COLOR_BGR2GRAY)
      img_arr_data = np.array(img_data_cvt)
      img_arr_data = img_arr_data.astype('float32')
      img_arr_data = np.expand_dims(img_arr_data, axis=2)

      if input_details[0]['dtype'] == np.int8:
        input_scale, input_zero_point = input_details[0]["quantization"]
        test_image = img_arr_data / input_scale + input_zero_point

      test_image = np.expand_dims(test_image, axis=0).astype(input_details[0]["dtype"])
      interpreter.set_tensor(input_details[0]['index'], test_image)
      interpreter.invoke()
      output = interpreter.get_tensor(output_details[0]["index"])[0]

      if path.split('/')[-1] == "face":
        if output[0] > output[1]:
          detect_face[0] += 1
        else:
          detect_face[1] += 1
      elif path.split('/')[-1] == "no_face":
        if output[0] < output[1]:
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
  model, input_model, output_model = load_model(path_tflite)
  predict(model, input_model, output_model, list_path_data_test)

if __name__ == "__main__":
    main()
