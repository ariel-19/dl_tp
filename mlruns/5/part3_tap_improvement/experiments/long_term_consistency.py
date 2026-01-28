# tp5/part3_tap_improvement/experiments/long_term_consistency.py
import os
import mlflow
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.optimizers import Adam
from ..models.tap_improved import ImprovedTemporalModel
from ..utils.data_loader import load_moving_mnist  # À implémenter

def train():
    # Paramètres
    batch_size = 32
    num_frames = 16
    img_size = 28
    latent_dim = 128
    num_heads = 4
    epochs = 100
    
    # Charger les données
    print("Chargement des données...")
    (train_data, _), (val_data, _) = load_moving_mnist(
        seq_len=num_frames,
        image_size=img_size
    )
    
    # Normalisation
    train_data = train_data.astype('float32') / 255.0
    val_data = val_data.astype('float32') / 255.0
    
    # Configuration MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("improved_tap")
    
    with mlflow.start_run():
        # Enregistrer les paramètres
        mlflow.log_params({
            "batch_size": batch_size,
            "num_frames": num_frames,
            "latent_dim": latent_dim,
            "num_heads": num_heads,
            "epochs": epochs
        })
        
        # Créer le modèle
        print("Création du modèle...")
        model = ImprovedTemporalModel(
            latent_dim=latent_dim,
            num_frames=num_frames,
            num_heads=num_heads
        )
        
        # Compiler le modèle
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Entraînement
        print("Début de l'entraînement...")
        history = model.fit(
            train_data,  # Entrée
            train_data,  # Cible (reconstruction)
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(val_data, val_data),
            callbacks=callbacks
        )
        
        # Enregistrer le modèle
        mlflow.tensorflow.log_model(
            model,
            "model",
            registered_model_name="improved_tap"
        )
        
        print("Entraînement terminé !")
        
        return model, history

if __name__ == "__main__":
    model, history = train()