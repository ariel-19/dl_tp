# tp5/part2_seq2seq/train_seq2seq.py
import os
import mlflow
import numpy as np
from datetime import datetime
from tensorflow.keras.optimizers import Adam
from .data_utils import generate_synthetic_sequence
from .seq2seq_model import build_seq2seq_attention

def prepare_sequences(X, y, max_encoder_seq_length, max_decoder_seq_length, output_vocab_size=1):
    """Prépare les séquences pour l'entrée du modèle."""
    # Créer les entrées du décodeur (décalées d'un pas de temps)
    decoder_input_data = np.zeros_like(y)
    decoder_input_data[:, 1:, :] = y[:, :-1, :]
    decoder_input_data[:, 0, :] = 0  # Marqueur de début de séquence
    
    # Créer les sorties du décodeur (one-hot encodées)
    decoder_target_data = np.zeros((y.shape[0], y.shape[1], output_vocab_size))
    decoder_target_data[:, :, 0] = y[:, :, 0]  # Pour la régression
    
    return {
        'encoder_input': X,
        'decoder_input': decoder_input_data,
        'decoder_target': decoder_target_data
    }

def train():
    # Paramètres
    max_encoder_seq_length = 100
    max_decoder_seq_length = 100
    latent_dim = 64
    batch_size = 32
    epochs = 50
    
    # Initialiser MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("seq2seq_attention")
    
    with mlflow.start_run():
        # Enregistrer les paramètres
        mlflow.log_params({
            "latent_dim": latent_dim,
            "batch_size": batch_size,
            "epochs": epochs,
            "max_encoder_seq_length": max_encoder_seq_length,
            "max_decoder_seq_length": max_decoder_seq_length
        })
        
        print("Génération des données...")
        X_train, y_train = generate_synthetic_sequence(
            n_samples=1000, 
            seq_length=max_encoder_seq_length
        )
        X_val, y_val = generate_synthetic_sequence(
            n_samples=200, 
            seq_length=max_encoder_seq_length
        )
        
        # Préparer les données
        train_data = prepare_sequences(X_train, y_train, max_encoder_seq_length, max_decoder_seq_length)
        val_data = prepare_sequences(X_val, y_val, max_encoder_seq_length, max_decoder_seq_length)
        
        # Construire le modèle
        print("Construction du modèle...")
        model = build_seq2seq_attention(
            input_vocab_size=1,  # Une seule caractéristique par pas de temps
            output_vocab_size=1,  # Une seule sortie par pas de temps
            latent_dim=latent_dim,
            max_encoder_seq_length=max_encoder_seq_length,
            max_decoder_seq_length=max_decoder_seq_length
        )
        
        # Compiler le modèle
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',  # Erreur quadratique moyenne pour la régression
            metrics=['mae']  # Erreur absolue moyenne
        )
        
        # Callback pour MLflow
        mlflow.tensorflow.autolog()
        
        # Entraînement
        print("Début de l'entraînement...")
        history = model.fit(
            [train_data['encoder_input'], train_data['decoder_input']],
            train_data['decoder_target'],
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(
                [val_data['encoder_input'], val_data['decoder_input']],
                val_data['decoder_target']
            ),
            verbose=1
        )
        
        # Enregistrer le modèle
        mlflow.tensorflow.log_model(
            model, 
            "model",
            registered_model_name="seq2seq_attention"
        )
        
        print("Entraînement terminé !")
        
        return model, history

if __name__ == "__main__":
    model, history = train()