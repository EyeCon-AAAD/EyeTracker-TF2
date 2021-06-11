import main as m
import tensorflow as tf
import numpy as np
import time


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


def get_tflite_test_scores(data):
    eye_test_left, eye_test_right, face_test, face_mask_test, y_test = data

    # Load TFLite Model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path="gaze_predictor_model.tflite")
    interpreter.allocate_tensors()

    # get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    """
        Left Eye: placeholder
        Right Eye: placeholder_1
        Face: face
        Face Mask: placeholder_2
        Output: Identity
    """

    print(f'Input Details : {input_details}\n'
          f'Output Details: {output_details}')

    # calculate MAE & STD
    err_x = []
    err_y = []

    print('Predicting samples...')
    for data_index in range(eye_test_left.shape[0]):
        # get one sample
        eye_left_data = eye_test_left[data_index]
        eye_right_data = eye_test_right[data_index]
        face_data = face_test[data_index]
        face_mask_data = face_mask_test[data_index]

        # reshape to expected input shape
        # nparr.reshape((1, nparr.shape[0], nparr.shape[1], nparr.shape[2]))
        eye_left_data = eye_left_data.reshape(
            (1, eye_left_data.shape[0], eye_left_data.shape[1], eye_left_data.shape[2]))
        eye_right_data = eye_right_data.reshape(
            (1, eye_right_data.shape[0], eye_right_data.shape[1], eye_right_data.shape[2]))
        face_data = face_data.reshape(
            (1, face_data.shape[0], face_data.shape[1], face_data.shape[2]))
        face_mask_data = face_mask_data.reshape(
            (1, face_mask_data.shape[0]))

        # set the input tensor with appropriate data
        interpreter.set_tensor(input_details[0]['index'], eye_left_data)
        interpreter.set_tensor(input_details[1]['index'], eye_right_data)
        interpreter.set_tensor(input_details[2]['index'], face_data)
        interpreter.set_tensor(input_details[3]['index'], face_mask_data)

        # predict/infer
        tic = time.perf_counter()
        interpreter.invoke()
        toc = time.perf_counter()

        print(f'Inferred index {data_index} in {toc - tic:0.4f} seconds')
        # get output
        output_data = interpreter.tensor(output_details[0]['index'])
        err_x.append(abs(output_data()[0][0] - y_test[data_index][0]))
        err_y.append(abs(output_data()[0][1] - y_test[data_index][1]))

    # mean absolute error
    mae_x = np.mean(err_x)
    mae_y = np.mean(err_y)

    # standard deviation
    std_x = np.std(err_x)
    std_y = np.std(err_y)

    # final results
    print(f'Results:\n'
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

    # get_test_scores(val_data)
    get_tflite_test_scores(val_data)


if __name__ == '__main__':
    main()