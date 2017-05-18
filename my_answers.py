import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
import re


# use numpy to transform 1d array into sliding windows of specified size
# http://stackoverflow.com/questions/35221645/rolling-window-on-a-circular-array/35221942
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

# turn array of sliding windows into outputs
def rolling_window_output(a):
    return [[row[-1]] for row  in a]
    


# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # transform the series into rows of sliding windows
    rolling_windows = rolling_window(series, window_size)
    X = rolling_windows[:-1]
    # turn sliding windows into output for each pair
    y = rolling_window_output(rolling_windows)[1:]
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    # Initialize model
    model = Sequential()
    # Specified layers from instructions
    model.add(LSTM(212, input_shape=(window_size,len_chars)))
    model.add(Dense(len_chars, activation='softmax'))


    # initialize optimizer
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile model --> make sure initialized optimizer and callbacks - as defined above - are used
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model


### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # find all unique characters in the text
    # remove as many non-english characters and character sequences as you can 
    alpha_num_punc = re.compile("[^\x00-\x7f]|\p")
    text = re.sub(alpha_num_punc, '', text)

# use numpy to transform 1d array into sliding windows of specified size
# http://stackoverflow.com/questions/40084931/taking-subarrays-from-numpy-array-with-given-stride-stepsize/40085052#40085052
def rolling_window_steps(a, L, S):
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # Convert text to list for window 
    series = np.array(list(text))
    # Get rolling window based on specified params
    inputs = rolling_window_steps(series, window_size, step_size)
    # Get rolling window for outputs based on specific params
    rolling_windows_outputs = rolling_window_steps(series[1:], window_size, step_size)
    # Map first character of each window for outputs
    outputs = [row[-1] for row in rolling_windows_outputs]
    return inputs,outputs