import tensorflow as tf
from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb


def convert():
    saved_model_path = 'gaze_prediction_model.h5'
    gaze_pred_model = tf.keras.models.load_model(saved_model_path)

    # some layer names have been renamed to:
    """
        Left Eye: placeholder
        Right Eye: placeholder_1
        Face: face
        Face Mask: placeholder_2
    """
    # Creates model info.
    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = "EyeTracker Gaze Prediction Model"
    model_meta.description = "Predicts the coordinates relative to the camera position of the phone camera"
    model_meta.version = "v1"
    model_meta.author = "EyeCon"

    # Create Input Info.
    input_left_eye_meta = _metadata_fb.TensorMetadataT()
    input_right_eye_meta = _metadata_fb.TensorMetadataT()
    input_face = _metadata_fb.TensorMetadataT()
    input_face_mask = _metadata_fb.TensorMetadataT()

    converter = tf.lite.TFLiteConverter.from_keras_model(gaze_pred_model)
    tflite_model = converter.convert()

    # save the model
    with open('gaze_predictor_model.tflite', 'wb') as f:
        f.write(tflite_model)


def main():
    # convert()
    inspect_model()


def inspect_model():
    interpreter = tf.lite.Interpreter(model_path="gaze_predictor_model.tflite")
    interpreter.allocate_tensors()

    # Print input shape and type
    inputs = interpreter.get_input_details()
    print('{} input(s):'.format(len(inputs)))
    for i in range(0, len(inputs)):
        print('{} {}'.format(inputs[i]['shape'], inputs[i]['dtype']))

    # Print output shape and type
    outputs = interpreter.get_output_details()
    print('\n{} output(s):'.format(len(outputs)))
    for i in range(0, len(outputs)):
        print('{} {}'.format(outputs[i]['shape'], outputs[i]['dtype']))


if __name__ == '__main__':
    main()
