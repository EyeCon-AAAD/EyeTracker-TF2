import numpy as np
import time


def load_data():
    filenames = ['train_eye_left.npy',
                 'train_eye_right.npy',
                 'train_face.npy',
                 'train_face_mask.npy',
                 'train_y.npy',
                 'val_eye_left.npy',
                 'val_eye_right.npy',
                 'val_face.npy',
                 'val_face_mask.npy',
                 'val_y.npy']

    # load files
    numpy_arrays = dict()

    for filename in filenames:
        # read numpy array
        tic = time.perf_counter()
        data = np.load(filename)
        toc = time.perf_counter()
        print(f'Loaded {filename} in {toc - tic:0.4f} seconds')
        numpy_arrays[filename] = data
        # print(f'{filename} array summary: \n'
        #      f'{numpy_arrays[filename]}')

    return numpy_arrays
