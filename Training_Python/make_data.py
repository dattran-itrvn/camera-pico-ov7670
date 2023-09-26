import os
import shutil

# list_path_data_no_face = ['/home/ttb/Downloads/raw-img/cane',
#                           '/home/ttb/Downloads/raw-img/cavallo',
#                           '/home/ttb/Downloads/raw-img/elefante',
#                           '/home/ttb/Downloads/raw-img/farfalla',
#                           '/home/ttb/Downloads/raw-img/gatto',
#                           '/home/ttb/Downloads/raw-img/gallina',
#                           '/home/ttb/Downloads/raw-img/pecora',
#                           '/home/ttb/Downloads/raw-img/mucca',
#                           '/home/ttb/Downloads/raw-img/ragno',
#                           '/home/ttb/Downloads/raw-img/scoiattolo',
#                           ]
path_data_no_face = '/home/ttb/Downloads/coco2017/test2017'
path_data_face = '/home/ttb/Downloads/img_align_celeba'

path_data_TRAIN_no_face_new = '/home/ttb/data_face_detect/no_face'
path_data_TRAIN_face_new = '/home/ttb/data_face_detect/face'
path_data_TEST_no_face_new = '/home/ttb/data_test/no_face'
path_data_TEST_face_new = '/home/ttb/data_test/face'

def cp_data(mode):
    if mode == 'train':
        # if os.path.exists(path_data_new) is None:
        shutil.rmtree(path_data_TRAIN_face_new)
        os.mkdir(path_data_TRAIN_face_new)
        shutil.rmtree(path_data_TRAIN_no_face_new)
        os.mkdir(path_data_TRAIN_no_face_new)

        for img_path in os.listdir(path_data_face)[:5500]:
            file_old = path_data_face + "/" + img_path
            file_new = path_data_TRAIN_face_new + "/" + img_path
            shutil.copy(file_old, file_new)

        print("Copy image have face OK")

        # for path in list_path_data_no_face:
        for path in os.listdir(path_data_no_face)[:5500]:
            file_old = path_data_no_face + "/" + path
            file_new = path_data_TRAIN_no_face_new + "/" + path
            shutil.copy(file_old, file_new)

        print("Copy image no face OK")

    elif mode == 'test':
        # if os.path.exists(path_data_new) is None:
        shutil.rmtree(path_data_TEST_face_new)
        os.mkdir(path_data_TEST_face_new)
        shutil.rmtree(path_data_TEST_no_face_new)
        os.mkdir(path_data_TEST_no_face_new)

        for img_path in os.listdir(path_data_face)[5500:5700]:
            file_old = path_data_face + "/" + img_path
            file_new = path_data_TEST_face_new + "/" + img_path
            shutil.copy(file_old, file_new)

        print("Copy image have face OK")

        for path in os.listdir(path_data_no_face)[5500:5700]:
            file_old = path_data_no_face + "/" + path
            file_new = path_data_TEST_no_face_new + "/" + path
            shutil.copy(file_old, file_new)

        print("Copy image have no face OK")

def main():
    cp_data('train')
    cp_data('test')

if __name__ == "__main__":
    main()