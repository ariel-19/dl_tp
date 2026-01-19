import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
import mlflow
import mlflow.tensorflow

# ======================================================
# 1. DATA PREPARATION
# ======================================================
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train_full = x_train_full.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Manual Split
x_val = x_train_full[54000:]
y_val = y_train_full[54000:]
x_train = x_train_full[:54000]
y_train = y_train_full[:54000]

# ======================================================
# 2. MODEL WITH BATCH NORMALIZATION
# ======================================================
model = keras.Sequential([
    keras.Input(shape=(784,)),
    
    # Dense Layer (with L2 form Ex 2)
    keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    
    # EXERCISE 4: Batch Normalization Layer
    # Placed between the Dense layer and the Dropout layer
    keras.layers.BatchNormalization(),
    
    # Dropout (from Ex 2)
    keras.layers.Dropout(0.2),
    
    keras.layers.Dense(10, activation='softmax')
])

# We use Adam (the winner from Ex 3)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ======================================================
# 3. TRAINING
# ======================================================
EPOCHS = 5
BATCH_SIZE = 128

mlflow.set_experiment("Exercise_4_Batch_Normalization")

with mlflow.start_run(run_name="Batch_Norm_Model"):
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("architecture", "Dense -> BatchNorm -> Dropout -> Output")

    print("\nStarting training with Batch Normalization...")
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, y_val),
        verbose=1
    )

    # Evaluation
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nResult (Batch Norm) - Test Acc: {test_acc:.4f}")

    # Log Metrics
    mlflow.log_metric("val_accuracy", history.history['val_accuracy'][-1])
    mlflow.log_metric("test_accuracy", test_acc)
    
    mlflow.tensorflow.log_model(model, "model")