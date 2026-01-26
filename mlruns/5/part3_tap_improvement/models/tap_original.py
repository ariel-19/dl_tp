# tp5/part3_tap_improvement/models/tap_original.py
import tensorflow as tf
from tensorflow.keras import layers, Model

class TemporalLatentModel(Model):
    def __init__(self, latent_dim=64, num_frames=16, **kwargs):
        super(TemporalLatentModel, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_frames = num_frames
        
        # Encodeur pour chaque frame
        self.encoder = tf.keras.Sequential([
            layers.Conv2D(32, 3, activation='relu', padding='same'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation='relu', padding='same'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu')
        ])
        
        # Modèle temporel (GRU)
        self.temporal_model = layers.GRU(
            latent_dim, 
            return_sequences=True,
            return_state=True
        )
        
        # Décodeur
        self.decoder = tf.keras.Sequential([
            layers.Dense(7*7*64, activation='relu'),
            layers.Reshape((7, 7, 64)),
            layers.Conv2DTranspose(64, 3, activation='relu', padding='same'),
            layers.UpSampling2D(),
            layers.Conv2DTranspose(32, 3, activation='relu', padding='same'),
            layers.UpSampling2D(),
            layers.Conv2D(1, 3, activation='sigmoid', padding='same')
        ])
    
    def call(self, inputs, training=False):
        # inputs shape: (batch_size, num_frames, H, W, C)
        batch_size = tf.shape(inputs)[0]
        
        # Encoder chaque frame
        encoded_frames = []
        for t in range(self.num_frames):
            frame = inputs[:, t, :, :, :]
            encoded = self.encoder(frame, training=training)
            encoded_frames.append(encoded)
        
        # Stack les représentations encodées
        # shape: (batch_size, num_frames, latent_dim)
        encoded_sequence = tf.stack(encoded_frames, axis=1)
        
        # Modélisation temporelle
        full_sequence, final_state = self.temporal_model(
            encoded_sequence, 
            training=training
        )
        
        # Décodage de chaque frame
        decoded_frames = []
        for t in range(self.num_frames):
            # Utiliser la sortie correspondante du GRU pour chaque pas de temps
            latent_vector = full_sequence[:, t, :]
            decoded = self.decoder(latent_vector, training=training)
            decoded_frames.append(decoded)
        
        # Stack les frames décodées
        # shape: (batch_size, num_frames, H, W, C)
        output_sequence = tf.stack(decoded_frames, axis=1)
        
        return output_sequence