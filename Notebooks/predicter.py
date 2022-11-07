"""
Class which predicts signals based on the previous inputs
Coding utf8
"""
import tensorflow as tf # For regular keras models
import tflite_runtime.interpreter as tflite # For TFLite models
import numpy as np

class Predicter:
    """
    Predicter class utilizing tflite models
    """
    def __init__(self, path, bounds = (-.2, .2)):
        """
        Constructor. Sets up normalization values and TFLite
        """
        self.bounds = bounds

        # tflite setup
        #self.model = tf.lite.Interpreter(model_path = path) # use this for regular tflite
        self.model = tflite.Interpreter(model_path = path) 
        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()

        
    def normalize(self, input):
        """
        Normalizes input signals based on normalization bounds
        """
        # Normalize by bounds
        in_norm = (input - self.bounds[0])/(self.bounds[1] - self.bounds[0])
        # Ensure signal range is between [0,1]
        in_norm[in_norm > 1] = 1
        in_norm[in_norm < 0] = 0

        return in_norm

    def denormalize(self, input):
        """
        Denormalize signal based on set bounds after prediction
        """
        return input * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
    
    def predict(self, input):
        """
        Predict future motor position based on the previous input
        """
        # Assume (30,2) input
        # Normalize input
        in_normalized = self.normalize(input).astype(np.float32)
        # tflite
        # set input and predict
        self.model.set_tensor(self.input_details[0]['index'], in_normalized[None, :,:])
        self.model.invoke()
        prediction = self.model.get_tensor(self.output_details[0]['index'])
        
        # Reset LSTM initial states
        self.model.reset_all_variables()
        # denormalize prediction
        prediction = self.denormalize(prediction)
        return prediction.squeeze()


class Predicter_norm:
    """
    Predicter class utilizing regular Keras models
    """
    def __init__(self, path, bounds = (-.2, .2)):
        """
        Constructor. Sets up normalization values and TFLite
        """
        self.bounds = bounds
        self.model = tf.keras.models.load_model(path)
        #self.model = model
        
    def normalize(self, input):
        """
        Normalizes input signals based on normalization bounds
        """
        # Normalize by bounds
        in_norm = (input - self.bounds[0])/(self.bounds[1] - self.bounds[0])
        # Ensure signal range is between [0,1]
        in_norm[in_norm > 1] = 1
        in_norm[in_norm < 0] = 0

        return in_norm

    def denormalize(self, input):
        """
        Denormalize signal based on set bounds after prediction
        """
        return input * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
    
    def predict(self, input):
        """
        Predict future motor position based on the previous input
        """
        
        # Normalize input
        in_normalized = self.normalize(input)

        prediction = self.model.predict(in_normalized[None,:,:])

        # denormalize prediction
        prediction = self.denormalize(prediction)
        return prediction.squeeze()