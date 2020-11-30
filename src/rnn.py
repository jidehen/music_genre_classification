import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess import get_data, get_batch

class Model(tf.keras.Model):
    def __init__(self):
        """
        The Model class predicts the genre of a track.
        """
        super(Model, self).__init__()

        self.batch_size = 75  
        self.rnn_size = 256
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.num_classes = 16
        self.LSTM = tf.keras.layers.LSTM(self.rnn_size, return_sequences=False, return_state=True)
        self.dense = tf.keras.layers.Dense(self.num_classes, activation='softmax')

    def call(self, inputs, initial_state=None):
        """
        :param inputs: shape [batch_size, time_steps, features]
        """        
        lstm_output, state1, state2  = self.LSTM(inputs, initial_state=initial_state)
        outputs = self.dense(lstm_output)
        return outputs

    def loss(self, probs, labels):
        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, probs))

    def accuracy(self, probabilities, labels):
        """
        Calculates the model's accuracy by comparing the number 
        of correct predictions with the correct answers.
        :param probabilities: result of running model.call() on test inputs
        :param labels: test set labels
        :return: Float (0,1) that contains batch accuracy
        """
        # TODO: calculate the batch accuracy
        count = 0
        for x in range(len(probabilities)):
             if np.argmax(labels[x]) == np.argmax(probabilities[x]):
                count += 1
        return count/len(probabilities) 