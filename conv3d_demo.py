import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import mlflow
import mlflow.tensorflow
import numpy as np
import json
import os

# ==========================================
# EXERCISE 3: CONV3D BLOCK IMPLEMENTATION
# ==========================================
def simple_conv3d_block(input_shape=(32, 32, 32, 1)):
    """
    Simple 3D CNN block for volumetric data processing.
    Input Shape: (Batch, Depth, Height, Width, Channels)
    """
    inputs = keras.Input(input_shape)

    # --- Block 1 ---
    # 16 filters, 3x3x3 kernel
    x = layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPool3D((2, 2, 2))(x)

    # --- Block 2 (TODO Completed) ---
    # 32 filters, 3x3x3 kernel
    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    # Downsampling in 3D
    x = layers.MaxPool3D((2, 2, 2))(x)

    # --- Classification Head ---
    x = layers.Flatten()(x)
    # Dummy output for binary classification (e.g., Tumor Present/Absent)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs, name="Simple_3D_CNN")
    return model

# ==========================================
# MAIN EXECUTION WITH MLFLOW
# ==========================================
if __name__ == '__main__':
    # 1. Setup MLflow Experiment
    mlflow.set_experiment("TP4_3D_Volumetric_Analysis")

    print("Starting Conv3D Experiment...")
    
    with mlflow.start_run(run_name="Conv3D_Baseline"):
        # 2. Build Model
        model_3d = simple_conv3d_block()
        model_3d.summary()

        # 3. Log Architecture (Engineering Practice)
        # We save the model configuration (layers, shapes) to a JSON file
        model_config = model_3d.to_json()
        
        # Create artifacts directory if it doesn't exist
        os.makedirs("artifacts", exist_ok=True)
        
        # Save config to file
        config_path = "artifacts/model_architecture.json"
        with open(config_path, "w") as f:
            f.write(model_config)
        
        # Log the file to MLflow
        mlflow.log_artifact(config_path)
        print(f"Model architecture saved to {config_path} and logged to MLflow.")

        # 4. Log Hyperparameters
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("filters_start", 16)
        mlflow.log_param("kernel_size", "(3,3,3)")
        
        # 5. Simulate Training Metrics (TODO Completed)
        # Since we don't have a real 3D dataset loaded, we simulate a result
        # to demonstrate the logging capability.
        print("Simulating training run...")
        dummy_final_val_loss = 0.345
        dummy_final_accuracy = 0.89
        
        mlflow.log_metric("final_val_loss", dummy_final_val_loss)
        mlflow.log_metric("final_accuracy", dummy_final_accuracy)

        print("MLflow tracking complete.")