import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
import matplotlib.pyplot as plt

class LSTM:
    def __init__(self, X_train_raw, X_test_raw,window=100, stride = 20, telescope=200):
        self.X_train_raw = X_train_raw
        self.X_test_raw = X_test_raw

        self.window = window
        self.stride = stride
        self.telescope = telescope
        
        self.X_train, self.y_train = self.build_sequences(self.X_train_raw, self.window, self.stride, self.telescope)
        self.X_test, self.y_test = self.build_sequences(self.X_test_raw, self.window, self.stride, self.telescope)       
        
        self.model= self.build_LSTM_model(input_shape=self.X_train.shape[1:], output_shape=self.y_train.shape[1:])
        
    def build_sequences(self,df, window=200, stride=20, telescope=100):
        # Sanity check to avoid runtime errors
        assert window % stride == 0
        dataset = []
        labels = []
        temp_df = df.copy().values
        temp_label = df.copy().values
        padding_check = len(df)%window

        if(padding_check != 0):
            # Compute padding length
            padding_len = window - len(df)%window
            padding = np.zeros((padding_len,temp_df.shape[1]), dtype='float32')
            temp_df = np.concatenate((padding,df))
            padding = np.zeros((padding_len,temp_label.shape[1]), dtype='float32')
            temp_label = np.concatenate((padding,temp_label))
            assert len(temp_df) % window == 0

        for idx in np.arange(0,len(temp_df)-window-telescope,stride):
            dataset.append(temp_df[idx:idx+window])
            labels.append(temp_label[idx+window:idx+window+telescope])

        dataset = np.array(dataset)
        labels = np.array(labels)
        return dataset, labels
    
    def build_CONV_LSTM_model(input_shape, output_shape):
        # Ensure the input time steps are at least as many as the output time steps
        assert input_shape[0] >= output_shape[0], "For this exercise we want input time steps to be >= of output time steps"

        # Define the input layer with the specified shape
        input_layer = tfkl.Input(shape=input_shape, name='input_layer')

        # Add a Bidirectional LSTM layer with 64 units
        x = tfkl.Bidirectional(tfkl.LSTM(64, return_sequences=True, name='lstm'), name='bidirectional_lstm')(input_layer)

        # Add a 1D Convolution layer with 128 filters and a kernel size of 3
        x = tfkl.Conv1D(128, 3, padding='same', activation='relu', name='conv')(x)

        # Add a final Convolution layer to match the desired output shape
        output_layer = tfkl.Conv1D(output_shape[1], 3, padding='same', name='output_layer')(x)

        # Calculate the size to crop from the output to match the output shape
        crop_size = output_layer.shape[1] - output_shape[0]

        # Crop the output to the desired length
        output_layer = tfkl.Cropping1D((0, crop_size), name='cropping')(output_layer)

        # Construct the model by connecting input and output layers
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name='CONV_LSTM_model')

        # Compile the model with Mean Squared Error loss and Adam optimizer
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam())

        return model
    
    def train(self, epochs=10, batch_size=32):
        history=self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        return history
    
    def plot_history(self, history):
        best_epoch = np.argmin(history['val_loss'])
        plt.figure(figsize=(17,4))
        plt.plot(history['loss'], label='Training loss', alpha=.8, color='#ff7f0e')
        plt.plot(history['val_loss'], label='Validation loss', alpha=.9, color='#5a9aa5')
        plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
        plt.title('Mean Squared Error')
        plt.legend()
        plt.grid(alpha=.3)
        plt.show()

        plt.figure(figsize=(18,3))
        plt.plot(history['lr'], label='Learning Rate', alpha=.8, color='#ff7f0e')
        plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
        plt.legend()
        plt.grid(alpha=.3)
        plt.show()
        
    def save_model(self, filename):
        self.model.save(filename)
        
    def load_model(self, filename):
        self.model = tfk.models.load_model(filename)
        
    def evaluate(self):
        predictions = self.model.predict(self.X_test, verbose=0)

        # Print the shape of the predictions
        print(f"Predictions shape: {predictions.shape}")

        # Calculate and print Mean Squared Error (MSE)
        mean_squared_error = tfk.metrics.mean_squared_error(self.y_test.flatten(), predictions.flatten()).numpy()
        print(f"Mean Squared Error: {mean_squared_error}")

        # Calculate and print Mean Absolute Error (MAE)
        mean_absolute_error = tfk.metrics.mean_absolute_error(self.y_test.flatten(), predictions.flatten()).numpy()
        print(f"Mean Absolute Error: {mean_absolute_error}")
        