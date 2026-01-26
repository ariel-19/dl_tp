# tp5/part3_tap_improvement/models/attention.py
import tensorflow as tf
from tensorflow.keras import layers

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        
    def build(self, input_shape):
        self.query = layers.Dense(self.key_dim * self.num_heads)
        self.key = layers.Dense(self.key_dim * self.num_heads)
        self.value = layers.Dense(self.key_dim * self.num_heads)
        self.dense = layers.Dense(input_shape[-1])
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.key_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs, mask=None):
        batch_size = tf.shape(inputs)[0]
        
        # Projections linéaires
        q = self.query(inputs)  # (B, T, num_heads * key_dim)
        k = self.key(inputs)    # (B, T, num_heads * key_dim)
        v = self.value(inputs)  # (B, T, num_heads * key_dim)
        
        # Séparation des têtes
        q = self.split_heads(q, batch_size)  # (B, num_heads, T, key_dim)
        k = self.split_heads(k, batch_size)  # (B, num_heads, T, key_dim)
        v = self.split_heads(v, batch_size)  # (B, num_heads, T, key_dim)
        
        # Calcul des scores d'attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (B, num_heads, T, T)
        
        # Mise à l'échelle
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Masquage optionnel
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Softmax sur le dernier axe
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        # Application des poids d'attention aux valeurs
        output = tf.matmul(attention_weights, v)  # (B, num_heads, T, key_dim)
        
        # Concaténation des têtes
        output = tf.transpose(output, perm=[0, 2, 1, 3])  # (B, T, num_heads, key_dim)
        concat_attention = tf.reshape(
            output, 
            (batch_size, -1, self.num_heads * self.key_dim)
        )
        
        # Projection finale
        output = self.dense(concat_attention)  # (B, T, D)
        
        return output
        