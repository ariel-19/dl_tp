# tp5/part2_seq2seq/data_utils.py
import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_sequence(n_samples=1000, seq_length=100, n_sines=3, noise_level=0.1):
    """
    Génère des séquences temporelles synthétiques comme combinaison de sinusoïdes.
    
    Args:
        n_samples: Nombre d'échantillons à générer
        seq_length: Longueur de chaque séquence
        n_sines: Nombre de sinusoïdes à combiner
        noise_level: Niveau de bruit à ajouter
        
    Returns:
        X: Données d'entrée (batch_size, seq_length, 1)
        y: Données de sortie (batch_size, seq_length, 1)
    """
    t = np.linspace(0, 10, seq_length)
    X = np.zeros((n_samples, seq_length, 1))
    y = np.zeros((n_samples, seq_length, 1))
    
    for i in range(n_samples):
        # Générer des fréquences et phases aléatoires
        freqs = np.random.uniform(0.5, 2.0, n_sines)
        phases = np.random.uniform(0, 2*np.pi, n_sines)
        amplitudes = np.random.uniform(0.1, 1.0, n_sines)
        
        # Créer la séquence comme combinaison de sinusoïdes
        signal = np.zeros(seq_length)
        for freq, phase, amp in zip(freqs, phases, amplitudes):
            signal += amp * np.sin(2 * np.pi * freq * t + phase)
        
        # Ajouter du bruit
        noise = np.random.normal(0, noise_level, seq_length)
        signal += noise
        
        # Créer une version décalée pour la prédiction
        shifted_signal = np.roll(signal, -1)
        shifted_signal[-1] = signal[-1]  # Répéter la dernière valeur
        
        X[i, :, 0] = signal
        y[i, :, 0] = shifted_signal
    
    return X, y

def plot_sequence(sequence, title="Séquence temporelle"):
    """Visualise une séquence temporelle."""
    plt.figure(figsize=(12, 4))
    plt.plot(sequence)
    plt.title(title)
    plt.xlabel("Temps")
    plt.ylabel("Valeur")
    plt.grid(True)
    plt.show()