import tensorflow as tf
import numpy as np
from tensorflow.keras import Model

class Model(tf.keras.Model):
    def __init__(self, vocab_size, actor_vocab_size):
        """
        The Model class predicts the genre of a track.
        """
        super(Model, self).__init__()

        self.batch_size = 150
        self.hidden_size = 100
        self.rnn_size = 256
        self.num_classes = 8
        self.vocab_size = vocab_size
        self.actor_vocab_size = actor_vocab_size
        self.embedding_size = 200
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.window_size = 250
        self.num_features = 14

        self.numerical_dense1 = tf.keras.layers.Dense(self.hidden_size)
        self.numerical_dense2 = tf.keras.layers.Dense(self.hidden_size)


        self.features_LSTM = tf.keras.layers.LSTM(self.rnn_size, input_shape=(self.window_size, self.num_features), return_sequences=True, return_state=True, dropout=.65, dtype=np.float64)
        self.features_LSTM2 = tf.keras.layers.LSTM(self.rnn_size, input_shape=(self.window_size, self.num_features), dropout=.65, dtype=np.float64)
        self.features_dense1 = tf.keras.layers.Dense(self.hidden_size)
        self.features_dense2 = tf.keras.layers.Dense(self.hidden_size)

        self.char_LSTM = tf.keras.layers.LSTM(self.rnn_size, return_state=True, dtype=np.float64)
        self.char_embeddings = tf.Variable(tf.random.truncated_normal(shape=[self.vocab_size, self.embedding_size], mean=0, stddev=0.01))
        self.char_dense1 = tf.keras.layers.Dense(self.hidden_size)
        self.char_dense2 = tf.keras.layers.Dense(self.hidden_size)

        self.actor_char_LSTM = tf.keras.layers.LSTM(self.rnn_size, return_state=True, dtype=np.float64)
        self.actor_char_embeddings = tf.Variable(tf.random.truncated_normal(shape=[self.actor_vocab_size, self.embedding_size], mean=0, stddev=0.01))
        self.actor_char_dense1 = tf.keras.layers.Dense(self.hidden_size)
        self.actor_char_dense2 = tf.keras.layers.Dense(self.hidden_size)

        self.dense_layer1 = tf.keras.layers.Dense(self.hidden_size)
        self.leaky1 = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.dense_layer2 = tf.keras.layers.Dense(self.hidden_size)
        self.leaky2 = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.dense_layer3 = tf.keras.layers.Dense(self.hidden_size)
        self.softmax_layer = tf.keras.layers.Dense(self.num_classes, activation='softmax')

    def call(self, numerical_inputs, feature_inputs, char_inputs, actor_char_inputs, initial_state=None, is_training=True):
        """
        :param inputs: shape [batch_size, features]
        """

        embeddings = tf.nn.embedding_lookup(self.char_embeddings, char_inputs)
        char_output, _, _ = self.char_LSTM(embeddings, initial_state=None)
        char_output = self.char_dense2(self.char_dense1(char_output))

        actor_embeddings = tf.nn.embedding_lookup(self.actor_char_embeddings, actor_char_inputs)
        actor_char_output, _, _ = self.actor_char_LSTM(actor_embeddings, initial_state=None)
        actor_char_output = self.actor_char_dense2(self.actor_char_dense1(actor_char_output))

        numerical_output = self.numerical_dense1(numerical_inputs)
        numerical_output = self.numerical_dense2(numerical_output)

        lstm_output, _, _ = self.features_LSTM(feature_inputs, initial_state=initial_state, training=is_training)
        lstm_output = self.features_LSTM2(lstm_output)
        lstm_output = self.features_dense2(self.features_dense1(lstm_output))

        inputs = tf.concat((lstm_output, numerical_output, actor_char_output, char_output), axis=1)

        return self.softmax_layer(self.dense_layer3(self.leaky2(self.dense_layer2(self.leaky1(self.dense_layer1(inputs))))))

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
