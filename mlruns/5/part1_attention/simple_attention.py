# tp5/part1_attention/simple_attention.py
import tensorflow as tf
from tensorflow.keras import layers

class SimpleAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch_size, seq_len, hidden_dim)
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform"
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros"
        )
        super(SimpleAttention, self).build(input_shape)

    def call(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        
        # Calculate attention scores
        # Shape: (batch_size, seq_len, 1)
        score = tf.tanh(tf.matmul(x, self.W) + self.b)
        
        # Attention weights
        # Shape: (batch_size, seq_len, 1)
        alignment_weights = tf.nn.softmax(score, axis=1)
        
        # Context vector
        # Shape: (batch_size, hidden_dim)
        context_vector = tf.reduce_sum(alignment_weights * x, axis=1)
        
        return context_vector, alignment_weights