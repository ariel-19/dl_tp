# tp5/part2_seq2seq/visualize.py
import numpy as np
import matplotlib.pyplot as plt
from .data_utils import generate_synthetic_sequence
from .seq2seq_model import build_seq2seq_attention

def visualize_predictions(model, n_examples=3, seq_length=100):
    """Visualise les prédictions du modèle sur des exemples de test."""
    # Générer des données de test
    X_test, y_test = generate_synthetic_sequence(
        n_samples=n_examples, 
        seq_length=seq_length,
        noise_level=0.05
    )
    
    # Préparer les entrées
    decoder_input = np.zeros_like(y_test)
    decoder_input[:, 1:, :] = y_test[:, :-1, :]
    
    # Faire des prédictions
    predictions = model.predict([X_test, decoder_input])
    
    # Afficher les résultats
    plt.figure(figsize=(15, 3 * n_examples))
    
    for i in range(n_examples):
        plt.subplot(n_examples, 1, i+1)
        plt.plot(X_test[i, :, 0], 'b-', label='Entrée')
        plt.plot(y_test[i, :, 0], 'g-', label='Vraie sortie')
        plt.plot(predictions[i, :, 0], 'r--', label='Prédiction')
        plt.title(f'Exemple {i+1}')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()