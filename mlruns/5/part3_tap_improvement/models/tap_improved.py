# tp5/part3_tap_improvement/models/tap_improved.py
import tensorflow as tf
from tensorflow.keras import layers, Model
from .attention import MultiHeadSelfAttention  # Nous allons l'implémenter ensuite

class ImprovedTemporalModel(Model):
    def __init__(self, latent_dim=128, num_frames=16, num_heads=4, **kwargs):
        super(ImprovedTemporalModel, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_frames = num_frames
        self.num_heads = num_heads
        
        # Encodeur plus profond avec résidus
        self.encoder = self._build_encoder()
        
        # Module d'auto-attention temporelle
        self.temporal_attention = MultiHeadSelfAttention(
            num_heads=num_heads,
            key_dim=latent_dim // num_heads
        )
        
        # Module de mémoire externe
        self.memory_slots = 8
        self.memory = layers.Dense(self.memory_slots * latent_dim)
        
        # Module de transition temporelle hiérarchique
        self.high_level_gru = layers.GRU(latent_dim, return_sequences=True)
        self.low_level_gru = layers.GRU(latent_dim, return_sequences=True)
        
        # Module de fusion
        self.fusion = layers.Dense(latent_dim, activation='tanh')
        
        # Décodeur amélioré
        self.decoder = self._build_decoder()
        
        # Couche de sortie
        self.output_conv = layers.Conv2D(1, 3, padding='same', activation='sigmoid')
    
    def _build_encoder(self):
        inputs = layers.Input(shape=(28, 28, 1))
        
        # Bloc 1
        x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.2)(x)
        
        # Bloc 2
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.2)(x)
        
        # Sortie
        x = layers.Flatten()(x)
        x = layers.Dense(self.latent_dim, activation='relu')(x)
        
        return tf.keras.Model(inputs=inputs, outputs=x)
    
    def _build_decoder(self):
        inputs = layers.Input(shape=(self.latent_dim,))
        
        x = layers.Dense(7*7*64, activation='relu')(inputs)
        x = layers.Reshape((7, 7, 64))(x)
        
        # Bloc 1
        x = layers.Conv2DTranspose(64, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D()(x)
        x = layers.Dropout(0.2)(x)
        
        # Bloc 2
        x = layers.Conv2DTranspose(32, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D()(x)
        x = layers.Dropout(0.2)(x)
        
        return tf.keras.Model(inputs=inputs, outputs=x)
    
    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        
        # Encoder chaque frame
        encoded_frames = []
        for t in range(self.num_frames):
            frame = inputs[:, t, :, :, :]
            encoded = self.encoder(frame, training=training)
            encoded_frames.append(encoded)
        
        # Stack les représentations encodées
        encoded_sequence = tf.stack(encoded_frames, axis=1)  # (B, T, D)
        
        # 1. Auto-attention temporelle
        attended_sequence = self.temporal_attention(encoded_sequence)
        
        # 2. Mémoire externe
        memory = self.memory(tf.reduce_mean(attended_sequence, axis=1))
        memory = tf.reshape(memory, [-1, self.memory_slots, self.latent_dim])
        
        # 3. Modélisation temporelle hiérarchique
        # Niveau haut (tendances globales)
        high_level = self.high_level_gru(attended_sequence)
        
        # Niveau bas (détails fins)
        low_level = self.low_level_gru(attended_sequence)
        
        # Fusion des représentations
        combined = tf.concat([high_level, low_level, memory], axis=-1)
        fused_representation = self.fusion(combined)
        
        # Décodage de chaque frame
        decoded_frames = []
        for t in range(self.num_frames):
            latent_vector = fused_representation[:, t, :]
            decoded = self.decoder(latent_vector, training=training)
            decoded = self.output_conv(decoded)
            decoded_frames.append(decoded)
        
        # Stack les frames décodées
        output_sequence = tf.stack(decoded_frames, axis=1)
        
        return output_sequence