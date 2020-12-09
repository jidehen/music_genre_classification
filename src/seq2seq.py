import numpy as np
import tensorflow as tf

class RNN_Seq2Seq(tf.keras.Model):
	def __init__(self, vocab_size):
		super(RNN_Seq2Seq, self).__init__()
		self.vocab_size = vocab_size
		self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

		self.batch_size = 100
		self.embedding_size = 150
		self.hidden_size = 264

		self.char_embeddings = tf.Variable(tf.random.truncated_normal(shape=[self.vocab_size, self.embedding_size], mean=0, stddev=0.01))
		self.GRU = tf.keras.layers.GRU(200, return_sequences=True, return_state=True)
		self.dense1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
		self.dense2 = tf.keras.layers.Dense(self.hidden_size, activation='softmax')

	def call(self, inputs):
		"""
		:param encoder_input: batched ids corresponding to french sentences
		:param decoder_input: batched ids corresponding to english sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
		"""

		gru_output, _ = self.GRU(tf.nn.embedding_lookup(self.char_embeddings, inputs), initial_state=None)
		output = self.dense2(self.dense1(gru_output[:, 0, :]))

		return output
