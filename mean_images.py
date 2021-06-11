import numpy as np

import main as m


def normalize(data, filename):
    shape = data.shape
    mean_data = np.reshape(data, (shape[0], -1))
    # scale to 0 - 1
    mean_data = mean_data.astype('float32') / 255.
    mean_data = np.mean(mean_data, axis=0)
    mean_data = np.reshape(mean_data, shape[1:])

    # save as binary file as export to java file. Written in little endian
    mean_data.tofile(filename + '.bin')


def prepare_data(data):
    eye_left, eye_right, face, face_mask, y = data
    # save mean_images of eye_left, eye_right, eye_face
    normalize(eye_left, 'mean_eye_left')
    normalize(eye_right, 'mean_eye_right')
    normalize(face, 'mean_face')


def main():
    print('Loading Data...')
    train_data, _ = m.get_train_val_data()
    #
    # normalized images
    print('Preparing Data...')
    # number of samples, added subsampling to try running or debug. None for all samples
    num_samples = None
    prepare_data(train_data)

    # read data
    mean_eye_left = np.fromfile('mean_eye_left.bin', np.float32)
    print(mean_eye_left)


if __name__ == '__main__':
    main()
