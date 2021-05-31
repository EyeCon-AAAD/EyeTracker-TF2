import main as m
import tensorflow as tf
import numpy as np


def get_test_scores(data):
    eye_test_left, eye_test_right, face_test, face_mask_test, y_test = data

    # load model
    model_name = 'gaze_prediction_model.h5'
    gaze_prediction_model = tf.keras.models.load_model(model_name)

    # evaluate model
    mse_loss = gaze_prediction_model.evaluate([eye_test_left, eye_test_right, face_test, face_mask_test], [y_test])
    predictions = gaze_prediction_model.predict(x=[eye_test_left, eye_test_right, face_test, face_mask_test],
                                                batch_size=128, verbose=1)

    # calculate MAE & STD
    err_x = []
    err_y = []

    for i, prediction in enumerate(predictions):
        err_x.append(abs(prediction[0] - y_test[i][0]))
        err_y.append(abs(prediction[1] - y_test[i][1]))

    # mean absolute error
    mae_x = np.mean(err_x)
    mae_y = np.mean(err_y)

    # standard deviation
    std_x = np.std(err_x)
    std_y = np.std(err_y)

    # final results
    print(f'Results:\n'
          f'MSE loss: {mse_loss}\n'
          f'MAE: {mae_x}, {mae_y}\n'
          f'STD: {std_x}, {std_y}')


def main():
    # load data
    print('Loading Data...')
    _, val_data = m.get_train_val_data()

    # normalized images
    print('Preparing Data...')
    # number of samples, added subsampling to try running or debug. None for all samples
    num_samples = None
    val_data = m.prepare_data(val_data, num_samples=num_samples)

    print('Data Prepared')

    get_test_scores(val_data)


if __name__ == '__main__':
    main()