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

        self.batch_size = 50
        self.rnn_size = 50
        self.hidden_size = 100 
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=.00001)
        #tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=.00001, decay_steps=10000, decay_rate=.9)
          #  ) #learning rate schedule: inverse time decay w/ floor (1e-5)
        self.num_classes = 8
        # transformer instead of lstm multiheaded
        # additional song features: 
        self.LSTM = tf.keras.layers.GRU(self.rnn_size, return_sequences=False, return_state=True, dropout=.3)
        self.leaky_relu = tf.keras.layers.LeakyReLU()
        self.dense_2 = tf.keras.layers.Dense(self.num_classes, activation='softmax')

    def call(self, inputs, initial_state=None):
        """
        :param inputs: shape [batch_size, time_steps, features]
        """        
        # pass thru dense layer
        lstm_output, state  = self.LSTM(inputs, initial_state=initial_state)
        outputs = self.leaky_relu(lstm_output)
        outputs = self.dense_2(outputs)
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
