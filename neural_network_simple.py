"""
Simple Feed-Forward Neural Network for Classification
====================================================

A standalone script demonstrating a simple neural network implementation using
TensorFlow/Keras for classification tasks. Includes data loading, architecture
design, training, and evaluation with visualization.

Author: AI Assistant
Date: January 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import warnings
import time
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_mnist_data():
    """Load and preprocess MNIST dataset."""
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Flatten images from 28x28 to 784
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    
    # Convert labels to categorical
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    return (x_train, y_train), (x_test, y_test), num_classes

def load_churn_data():
    """Load and preprocess churn dataset."""
    print("Loading churn dataset...")
    try:
        # Load the churn dataset
        df = pd.read_csv('Churn Prdiction Data/churn-bigml-80.csv')
        print(f"Dataset shape: {df.shape}")
        print(f"Dataset columns: {df.columns.tolist()}")
        
        # Display basic info
        print("\nDataset info:")
        print(df.info())
        print(f"\nMissing values:\n{df.isnull().sum()}")
        
        # Separate features and target
        target_col = 'Churn'
        if target_col not in df.columns:
            # Try alternative column names
            possible_targets = ['churn', 'Churn', 'target', 'Target', 'class', 'Class']
            for col in possible_targets:
                if col in df.columns:
                    target_col = col
                    break
            else:
                # Use the last column as target
                target_col = df.columns[-1]
                print(f"Using '{target_col}' as target column")
        
        # Prepare features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle target variable
        if y.dtype == 'object' or y.dtype == 'bool':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
        
        # Convert to numpy arrays
        X = X.values.astype('float32')
        y = y.values
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Determine number of classes
        num_classes = len(np.unique(y))
        
        if num_classes == 2:
            # Binary classification - use sigmoid activation
            y_train = y_train.astype('float32')
            y_test = y_test.astype('float32')
        else:
            # Multi-class classification - use categorical
            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Number of classes: {num_classes}")
        
        return (X_train, y_train), (X_test, y_test), num_classes
        
    except Exception as e:
        print(f"Error loading churn data: {e}")
        print("Falling back to MNIST dataset...")
        return load_mnist_data()

def create_neural_network(input_shape, num_classes, architecture='simple'):
    """
    Create a feed-forward neural network.
    
    Args:
        input_shape: Shape of input features
        num_classes: Number of output classes
        architecture: Network architecture type ('simple', 'medium', 'complex')
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential()
    
    if architecture == 'simple':
        # Simple architecture: Input -> Hidden(128) -> Hidden(64) -> Output
        model.add(layers.Dense(128, activation='relu', input_shape=(input_shape,)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.2))
        
    elif architecture == 'medium':
        # Medium architecture: Input -> Hidden(256) -> Hidden(128) -> Hidden(64) -> Output
        model.add(layers.Dense(256, activation='relu', input_shape=(input_shape,)))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.2))
        
    elif architecture == 'complex':
        # Complex architecture: Input -> Hidden(512) -> Hidden(256) -> Hidden(128) -> Hidden(64) -> Output
        model.add(layers.Dense(512, activation='relu', input_shape=(input_shape,)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.2))
    
    # Output layer
    if num_classes == 2:
        # Binary classification
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    else:
        # Multi-class classification
        model.add(layers.Dense(num_classes, activation='softmax'))
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return model

def plot_training_history(history):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training & validation accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot training & validation loss
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('neural_network_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, num_classes):
    """Plot confusion matrix."""
    if num_classes == 2:
        # Binary classification
        cm = confusion_matrix(y_true, (y_pred > 0.5).astype(int))
        labels = ['No Churn', 'Churn']
    else:
        # Multi-class classification
        y_true_labels = np.argmax(y_true, axis=1) if len(y_true.shape) > 1 else y_true
        y_pred_labels = np.argmax(y_pred, axis=1)
        cm = confusion_matrix(y_true_labels, y_pred_labels)
        labels = [f'Class {i}' for i in range(num_classes)]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('neural_network_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, X_test, y_test, num_classes):
    """Evaluate the trained model."""
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    if num_classes == 2:
        # Binary classification
        y_pred_binary = (y_pred > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred_binary)
        precision = precision_score(y_test, y_pred_binary)
        recall = recall_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_binary, 
                                  target_names=['No Churn', 'Churn']))
    else:
        # Multi-class classification
        y_true_labels = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
        y_pred_labels = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_true_labels, y_pred_labels)
        precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
        recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
        f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (weighted): {precision:.4f}")
        print(f"Recall (weighted): {recall:.4f}")
        print(f"F1-Score (weighted): {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true_labels, y_pred_labels))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, num_classes)
    
    return y_pred

def main():
    """Main execution function."""
    print("="*60)
    print("NEURAL NETWORK CLASSIFIER WITH TENSORFLOW/KERAS")
    print("="*60)
    
    # Choose dataset
    print("\nAvailable datasets:")
    print("1. MNIST (handwritten digits)")
    print("2. Churn Prediction Dataset")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == '1':
        (X_train, y_train), (X_test, y_test), num_classes = load_mnist_data()
        dataset_name = "MNIST"
    else:
        (X_train, y_train), (X_test, y_test), num_classes = load_churn_data()
        dataset_name = "Churn"
    
    # Choose architecture
    print("\nAvailable architectures:")
    print("1. Simple (128 -> 64)")
    print("2. Medium (256 -> 128 -> 64)")
    print("3. Complex (512 -> 256 -> 128 -> 64)")
    
    arch_choice = input("\nEnter your choice (1, 2, or 3): ").strip()
    architectures = {'1': 'simple', '2': 'medium', '3': 'complex'}
    architecture = architectures.get(arch_choice, 'simple')
    
    print(f"\nUsing {architecture} architecture for {dataset_name} dataset")
    
    # Create model
    input_shape = X_train.shape[1]
    model = create_neural_network(input_shape, num_classes, architecture)
    
    # Display model summary
    print("\n" + "="*50)
    print("MODEL ARCHITECTURE")
    print("="*50)
    model.summary()
    
    # Train model
    print("\n" + "="*50)
    print("TRAINING MODEL")
    print("="*50)
    
    # Set training parameters
    epochs = 50 if dataset_name == "MNIST" else 100
    batch_size = 128
    
    # Add callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
    ]
    
    print(f"Training for up to {epochs} epochs with batch size {batch_size}")
    start_time = time.time()
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    y_pred = evaluate_model(model, X_test, y_test, num_classes)
    
    # Save model
    model_filename = f'neural_network_{dataset_name.lower()}_{architecture}.h5'
    model.save(model_filename)
    print(f"\nModel saved as: {model_filename}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"Files generated:")
    print(f"- {model_filename} (trained model)")
    print(f"- neural_network_training_history.png")
    print(f"- neural_network_confusion_matrix.png")

if __name__ == "__main__":
    main()
