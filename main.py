import data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def get_train_val_data():
    data_dict = data.load_data()

    train_eye_left = data_dict['train_eye_left.npy']
    train_eye_right = data_dict['train_eye_right.npy']
    train_face = data_dict['train_face.npy']
    train_face_mask = data_dict['train_face_mask.npy']
    train_y = data_dict['train_y.npy']
    val_eye_left = data_dict['val_eye_left.npy']
    val_eye_right = data_dict['val_eye_right.npy']
    val_face = data_dict['val_face.npy']
    val_face_mask = data_dict['val_face_mask.npy']
    val_y = data_dict['val_y.npy']

    return [train_eye_left, train_eye_right, train_face, train_face_mask, train_y], \
           [val_eye_left, val_eye_right, val_face, val_face_mask, val_y]


def normalize(data):
    shape = data.shape
    normalised_data = np.reshape(data, (shape[0], -1))
    normalised_data = normalised_data.astype('float32') / 255.  # scaling
    normalised_data = normalised_data - np.mean(normalised_data, axis=0)  # normalizing
    return np.reshape(normalised_data, shape)


def prepare_data(data):
    eye_left, eye_right, face, face_mask, y = data
    eye_left = normalize(eye_left)
    eye_right = normalize(eye_right)
    face = normalize(face)
    face_mask = np.reshape(face_mask, (face_mask.shape[0], -1)).astype('float32')
    y = y.astype('float32')
    return [eye_left, eye_right, face, face_mask, y]


def train(model, train_data, val_data, batch_size=128, learning_rate=1e-3, epochs=1000):
    """
    Loss: mse-> Mean Squared Error
    :param model:
    :param train_data:
    :param val_data:
    :param learning_rate:
    :return:
    """
    eye_left_train = train_data[0]
    eye_right_train = train_data[1]
    face_train = train_data[2]
    face_mask_train = train_data[3]

    eye_left_val = val_data[0]
    eye_right_val = val_data[1]
    face_val = val_data[2]
    face_mask_val = val_data[3]

    y_train = train_data[4]
    y_val = val_data[4]

    model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.Adam(lr=learning_rate))

    # save the model with best performance on the validation set
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('gaze_prediction_model.h5', save_best_only=True)

    # perform early stopping when there's no increase in performance on the validation set
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

    history = model.fit([eye_left_train, eye_right_train, face_train, face_mask_train], [y_train],
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=([eye_left_val, eye_right_val, face_val, face_mask_val], [y_val]),
                        callbacks=[checkpoint_cb, early_stopping_cb])

    return history


def plot_training_metrics(history):
    pd.Dataframe(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


def create_model(train_data):
    # Convolutional Network Parameters
    image_size = 64
    num_channels = 3
    face_mask_size = 25

    # layers params for the left and right eye
    # CONV-E1
    conv_e1_kernel_size = 11
    conv_e1_kernel_number = 96
    max_pool_e1_kernel_size = 2  # strides defaults to 2 as well

    # CONV-E2
    conv_e2_kernel_size = 5
    conv_e2_kernel_number = 256
    max_pool_e2_kernel_size = 2  # strides defaults to 2 as well

    # CONV-E3
    conv_e3_kernel_size = 3
    conv_e3_kernel_number = 384
    max_pool_e3_kernel_size = 2  # strides defaults to 2 as well

    # CONV-E4
    conv_e4_kernel_size = 1
    conv_e4_kernel_number = 64
    max_pool_e4_kernel_size = 2  # strides defaults to 2 as well

    # to-do eye_size calculation from GitHub Repo

    # layers params for the face
    # CONV-F1
    conv_f1_kernel_size = 11
    conv_f1_kernel_number = 96
    max_pool_f1_kernel_size = 2  # strides defaults to 2 as well

    # CONV-F2
    conv_f2_kernel_size = 5
    conv_f2_kernel_number = 256
    max_pool_f2_kernel_size = 2  # strides defaults to 2 as well

    # CONV-F3
    conv_f3_kernel_size = 3
    conv_f3_kernel_number = 384
    max_pool_f3_kernel_size = 2  # strides defaults to 2 as well

    # CONV-F4
    conv_f4_kernel_size = 1
    conv_f4_kernel_number = 64
    max_pool_f4_kernel_size = 2  # strides defaults to 2 as well

    # to-do face_size calculation from GitHub Repo

    # Build the model using the Keras Functional API
    input_eye_left = tf.keras.layers.Input(shape=train_data[0].shape[1:], name='Left Eye')
    input_eye_right = tf.keras.layers.Input(shape=train_data[1].shape[1:], name='Right Eye')
    input_face = tf.keras.layers.Input(shape=train_data[2].shape[1:], name='Face')
    input_face_mask = tf.keras.layers.Input(shape=train_data[3].shape[1:], name='Face Mask')

    # ---------------------------------------- CONV-E1 -------------------------------------------------------
    conv_e1 = tf.keras.layers.Conv2D(filters=conv_e1_kernel_number, kernel_size=conv_e1_kernel_size, strides=1,
                                     padding='VALID', activation='relu', name='CONV_E1')
    # sharing weights
    conv_e1_left = conv_e1(input_eye_left)
    conv_e1_right = conv_e1(input_eye_right)

    conv_e1_max_pool = tf.keras.layers.MaxPool2D(max_pool_e1_kernel_size)
    conv_e1_max_pool_left = conv_e1_max_pool(conv_e1_left)
    conv_e1_max_pool_right = conv_e1_max_pool(conv_e1_right)

    # ---------------------------------------- CONV-E2 -------------------------------------------------------
    conv_e2 = tf.keras.layers.Conv2D(filters=conv_e2_kernel_number, kernel_size=conv_e2_kernel_size, strides=1,
                                     padding='VALID', activation='relu', name='CONV_E2')
    conv_e2_out_left = conv_e2(conv_e1_max_pool_left)
    conv_e2_out_right = conv_e2(conv_e1_max_pool_right)

    conv_e2_max_pool = tf.keras.layers.MaxPool2D(max_pool_e2_kernel_size)
    conv_e2_max_pool_left = conv_e2_max_pool(conv_e2_out_left)
    conv_e2_max_pool_right = conv_e2_max_pool(conv_e2_out_right)

    # ---------------------------------------- CONV-E3 -------------------------------------------------------
    conv_e3 = tf.keras.layers.Conv2D(filters=conv_e3_kernel_number, kernel_size=conv_e3_kernel_size, strides=1,
                                     padding='VALID', activation='relu', name='CONV_E3')
    conv_e3_out_left = conv_e3(conv_e2_max_pool_left)
    conv_e3_out_right = conv_e3(conv_e2_max_pool_right)

    conv_e3_max_pool = tf.keras.layers.MaxPool2D(max_pool_e3_kernel_size)
    conv_e3_max_pool_left = conv_e3_max_pool(conv_e3_out_left)
    conv_e3_max_pool_right = conv_e3_max_pool(conv_e3_out_right)

    # ---------------------------------------- CONV-E4 -------------------------------------------------------
    conv_e4 = tf.keras.layers.Conv2D(filters=conv_e4_kernel_number, kernel_size=conv_e4_kernel_size, strides=1,
                                     padding='VALID', activation='relu', name='CONV_E4')
    conv_e4_out_left = conv_e4(conv_e3_max_pool_left)
    conv_e4_out_right = conv_e4(conv_e3_max_pool_right)

    conv_e4_max_pool = tf.keras.layers.MaxPool2D(max_pool_e4_kernel_size)
    conv_e3_max_pool_left = conv_e4_max_pool(conv_e4_out_left)
    conv_e3_max_pool_right = conv_e4_max_pool(conv_e4_out_right)

    # ------------------------------------------ FC-E1 -------------------------------------------------------
    eye_concat = tf.keras.layers.Concatenate()([conv_e3_max_pool_left, conv_e3_max_pool_right])
    eye_concat = tf.keras.layers.Flatten()(eye_concat)
    fc_eye = tf.keras.layers.Dense(units=128, activation='relu', name='FC-E1')(eye_concat)

    # ---------------------------------------- CONV-F1 -------------------------------------------------------
    face = tf.keras.layers.Conv2D(filters=conv_f1_kernel_number, kernel_size=conv_f1_kernel_size, strides=1,
                                  padding='VALID', activation='relu', name='CONV_F1')
    face = face(input_face)
    face = tf.keras.layers.MaxPool2D(max_pool_f1_kernel_size)(face)

    # ---------------------------------------- CONV-F2 -------------------------------------------------------
    face = tf.keras.layers.Conv2D(filters=conv_f2_kernel_number, kernel_size=conv_f2_kernel_size, strides=1,
                                  padding='VALID', activation='relu', name='CONV_F2')(face)
    face = tf.keras.layers.MaxPool2D(max_pool_f2_kernel_size)(face)

    # ---------------------------------------- CONV-F3 -------------------------------------------------------
    face = tf.keras.layers.Conv2D(filters=conv_f3_kernel_number, kernel_size=conv_f3_kernel_size, strides=1,
                                  padding='VALID', activation='relu', name='CONV_F3')(face)
    face = tf.keras.layers.MaxPool2D(max_pool_f3_kernel_size)(face)

    # ---------------------------------------- CONV-F4 -------------------------------------------------------
    face = tf.keras.layers.Conv2D(filters=conv_f4_kernel_number, kernel_size=conv_f4_kernel_size, strides=1,
                                  padding='VALID', activation='relu', name='CONV_F4')(face)
    face = tf.keras.layers.MaxPool2D(max_pool_f4_kernel_size)(face)
    # ------------------------------------------ FC-F1 -------------------------------------------------------
    face = tf.keras.layers.Flatten()(face)
    fc_face = tf.keras.layers.Dense(units=128, activation='relu', name='FC-F1')(face)

    # ------------------------------------------ FC-FG1 -------------------------------------------------------
    face_mask = tf.keras.layers.Dense(units=256, activation='relu', name='FC-FG1')(input_face_mask)
    face_mask = tf.keras.layers.Flatten()(face_mask)
    # ------------------------------------------ FC-F2 -------------------------------------------------------
    face_face_mask = tf.keras.layers.Concatenate()([fc_face, face_mask])
    # not sure of the num_units
    face_face_mask = tf.keras.layers.Dense(units=128, activation='relu', name='FC-F2')(face_face_mask)

    # ------------------------------------------ FC-1 -------------------------------------------------------
    # concatenate Eyes with face
    fc = tf.keras.layers.Concatenate()([fc_eye, face_face_mask])
    fc = tf.keras.layers.Dense(units=128, activation='relu', name='FC-1')(fc)

    # ------------------------------------------ FC-2 -------------------------------------------------------
    output = tf.keras.layers.Dense(units=2, activation='relu', name='FC-2')(fc)

    model = tf.keras.Model(inputs=[input_eye_left, input_eye_right, input_face, input_face_mask],
                           outputs=[output])

    return model


def main():
    # load data
    train_data, val_data = get_train_val_data()
    gaze_prediction_model = create_model(train_data)

    # plot the model to confirm structure
    print(gaze_prediction_model.summary())
    tf.keras.utils.plot_model(gaze_prediction_model, 'Gaze-Prediction_Model.png', show_shapes=True,
                              show_layer_names=True)

    # normalized images
    train_data = prepare_data(train_data)
    val_data = prepare_data(val_data)

    # compile & train the model
    history = train(gaze_prediction_model, train_data, val_data, batch_size=128, learning_rate=1e-5, epochs=1000)

    # plot history
    plot_training_metrics(history)



if __name__ == '__main__':
    main()
