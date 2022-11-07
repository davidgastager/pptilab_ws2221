# Prediction Model Setup and Training
This folder contains all the necessary notebooks for preparing the notebooks, creating and training the model. As it is still a work in progress, not everything is working yet.

## Notebook 01_Dataset_Interpolation
This notebook deals with interpolating the data so it has uniform timesteps and saves them as dictionaries.

## Notebook 02_Dataset_Cleanup
Used to find and set the portions of the recorded datasets that will be used for training. The goal is to get rid of the parts where the ball fell off the platform (indicated by wobble in signal).

## Notebook 03_Data_Preparation
This notebook combines the recorded signals, splits all the data into a train/test/validation sets and windows them to specified dimensions. The final function was also put in a python file (__dataloader.py__) so it can be used easily in the subsequent notebooks.

## Notebook 04_Frequency_Analysis
Analysis of the input signals to reduce sampling frequency. 98.4% of the signals average energy is in the spectrum of 0-5 Hz. Therefore the sampling rate was adjusted to 10Hz.

## Notebook 05_Model
The model is built and trained in here. We tried 5 different models originating from 3 papers:
1. 'lmt_1' based on: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8633843 
2. 'lmt_2' based on: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8429079 (model a)
3. 3 models based on: https://www.researchgate.net/publication/354638949_Deep_Fusion_of_a_Skewed_Redundant_Magnetic_and_Inertial_Sensor_for_Heading_State_Estimation_in_a_Saturated_Indoor_Environment
- 'magnetic_a' based on model a) (simple LSTM)
- 'magnetic_b' based on model b) (subdivided LSTM)
- 'magnetic_c' based on model c) (dense LSTM)
<br>
We eventually tested everything using models 'lmt_1', 'lmt_2' and 'magnetic_a', as they yielded the best results.

## Notebook 06_TFLite_Conversion
The trained models are converted to TFLite for inference on the nanopis. This notebook handles the conversion of all models in the */models* directory.

These links were mainly used for reference:
1. https://www.tensorflow.org/lite/convert/rnn
2. https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/experimental_new_converter/Keras_LSTM_fusion_Codelab.ipynb#scrollTo=0-b0IKK2FGuO

## Notebook 07_Plot_Trainins:
The plots used for the report were generated here.

# Additional Files:
- __dataloader.py__: Wrapper class for easy access to the cleaned Data.
- __predicter.py__: Wrapper class to use the trained models.