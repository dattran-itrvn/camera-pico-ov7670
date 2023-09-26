import cv2
import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.optimizers import SGD
import tensorflow as tf
from matplotlib import pyplot as plt

class face_detect:
    def __init__(self, _obj = None, _num_class = 0):
        if _obj is None:
            _obj = []
        self.obj       = _obj
        self.num_class = _num_class
        self.labels    = None
        self.img_data = None
        self.model = None
        self.valid_images = [".jpg", ".png"]

    def load_data(self, path_data):
        labels = []
        img_data_list = []
        count = 0
        for index, obj in enumerate(self.obj):
            dir_path = path_data + '/' + obj
            for img_path in os.listdir(dir_path):
                name, ext = os.path.splitext(img_path)
                if ext.lower() not in self.valid_images:
                    continue
                self.img_data = cv2.imread(dir_path + '/' + img_path)
                self.img_data = cv2.resize(self.img_data, (60, 60))
                self.img_data = cv2.cvtColor(self.img_data, cv2.COLOR_BGR2GRAY)

                img_data_list.append(self.img_data)
                labels.append(index)
                count += 1
            print(f'{obj} : {count}')
            count = 0
        self.img_data = np.array(img_data_list)
        self.img_data = self.img_data.astype('float32')
        self.labels = np.array(labels ,dtype='int64')
            # scale down(so easy to work with)
        self.img_data = np.expand_dims(self.img_data, axis=3)
        print('Data loaded ... ')

    def train_model(self, _epoch, _path_save_model):
        epochs = _epoch
        l_rate = 0.01
        decay = l_rate/epochs

        # convert class labels to on-hot encoding
        _Y = np_utils.to_categorical(self.labels, self.num_class)
        # Shuffle the dataset
        # Split the dataset
        x, y = shuffle(self.img_data, _Y, random_state=2)
        _X_train, _X_test, _y_train, _y_test = train_test_split(x, y, test_size=0.2, random_state=2)
        input_shape=self.img_data[0].shape

        input_layer = tf.keras.Input(shape=input_shape)
        conv1 = tf.keras.layers.Conv2D(32,
                                       (3, 3),
                                        padding='same',
                                        activation='relu',
                                        kernel_constraint=maxnorm(3))(input_layer)
        max_pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv1)
        conv2 = tf.keras.layers.SeparableConv2D(64,
                                       (3, 3),
                                        padding='same',
                                        activation='relu',
                                        kernel_constraint=maxnorm(3))(max_pool1)
        max_pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv2)
        conv3 = tf.keras.layers.SeparableConv2D(64,
                                       (3, 3),
                                        padding='same',
                                        activation='relu',
                                        kernel_constraint=maxnorm(3))(max_pool2)
        max_pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv3)
        conv4 = tf.keras.layers.SeparableConv2D(128,
                                       (3, 3),
                                        padding='same',
                                        activation='relu',
                                        kernel_constraint=maxnorm(3))(max_pool3)
        average_pooling = tf.keras.layers.GlobalAveragePooling2D()(conv4)
        drop3 = tf.keras.layers.Dropout(0.5)(average_pooling)
        output_layer = tf.keras.layers.Dense(self.num_class, activation='softmax')(drop3)

        # Compile model
        sgd = SGD(lr=l_rate, momentum=0.9, decay=decay, nesterov=False)
        self.model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        print(self.model.summary())

        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        # Fit the model
        self.model.fit(_X_train, _y_train, validation_data=(_X_test, _y_test), epochs=epochs, batch_size=32)
        # Final evaluation of the model
        scores = self.model.evaluate(_X_test, _y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(_path_save_model + "/face.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(_path_save_model + "/face.h5")
        print("Saved model to disk")


    def representative_data(self, path_data_quantum):
        img_data_test_list = []
        # Load data quantum
        for index, obj in enumerate(self.obj):
            dir_path = path_data_quantum + '/' + obj
            for img_path in os.listdir(dir_path)[:10]:
                name, ext = os.path.splitext(img_path)
                if ext.lower() not in self.valid_images:
                    continue
                img_data_test = cv2.imread(dir_path + '/' + img_path)
                img_data_test = cv2.resize(img_data_test, (60, 60))
                img_data_test = cv2.cvtColor(img_data_test, cv2.COLOR_BGR2GRAY)
                img_data_test_list.append(img_data_test)

        img_data_test = np.array(img_data_test_list)
        img_data_test = img_data_test.astype('float32')
        # img_data_test = img_data_test/255.0
        # scale down(so easy to work with)
        img_data_test = np.expand_dims(img_data_test, axis=3)
        img_data_test = np.expand_dims(img_data_test, axis=0)

        return img_data_test

    def convert_h5_to_tflite(self, _path_model, _path_data_quantum):
        model = self.model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        data_quantum = self.representative_data(_path_data_quantum)

        def representative_data_gen():
            for input_value in tf.data.Dataset.from_tensor_slices(data_quantum).take(100):
                yield [input_value]

        converter.representative_dataset = representative_data_gen
        # Using Integer Quantization.
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Setting the input and output tensors to uint8.
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        # Converting the model.
        int_quant_model = converter.convert()

        # Saving the Integer Quantized TF Lite model.
        with open(_path_model + '/int_quant_model.tflite', 'wb') as f:
            f.write(int_quant_model)

def main():
    obj_t = ['face', 'no_face']
    num_classes: int = 2
    epochs: int = 15
    data_path = '/home/ttb/data_face_detect'
    model_path = '/home/ttb/PycharmProjects/OV7670_cam/model'

    face_detect_class = face_detect(obj_t, num_classes)
    face_detect_class.load_data(data_path)
    face_detect_class.train_model(epochs, model_path)
    face_detect_class.convert_h5_to_tflite(model_path, data_path)

if __name__ == "__main__":
    main()

