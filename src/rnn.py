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

        self.batch_size = 200
        self.rnn_size = 256
        self.hidden_size = 164 
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=.001)
        self.num_classes = 8
        self.window_size = 250
        self.num_features = 14
        # transformer instead of lstm multiheaded
        # additional song features: 
        self.LSTM = tf.keras.layers.LSTM(self.rnn_size, input_shape=(self.window_size, self.num_features), return_sequences=True, return_state=True, dropout=.65, dtype=np.float64)
        self.LSTM_2 = tf.keras.layers.LSTM(self.rnn_size, input_shape=(self.window_size, self.num_features), dropout=.65, dtype=np.float64)
        self.dense_1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        #self.dense_2 = tf.keras.layers.Dense(256, activation='relu')
        self.soft_max = tf.keras.layers.Dense(self.num_classes, activation='softmax')

    def call(self, inputs, initial_state=None, is_training=True):
        """
        :param inputs: shape [batch_size, time_steps, features]
        """        
        # pass thru dense layer
        outputs, _, _  = self.LSTM(inputs, initial_state=initial_state, training=is_training)
        outputs, _ = self.LSTM_2(outputs)
        outputs = self.dense_1(ouputs)
        #outputs = self.dropout(outputs)
        #outputs = self.dense_2(outputs)
        outputs = self.soft_max(outputs) 
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
        for x in range(np.shape(probabilities)[0]):
            if np.argmax(labels[x]) == np.argmax(probabilities[x]):
                count += 1
        return count/len(probabilities) 
