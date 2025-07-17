"""
Simple Feed-Forward Neural Network for MNIST Classification
===========================================================

A focused implementation of Task 3: Building a simple feed-forward neural network
using TensorFlow/Keras for MNIST digit classification, including data loading,
architecture design, training, and evaluation with visualization.

Author: AI Assistant
Date: January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set matplotlib style
plt.style.use('seaborn-v0_8')

def load_and_preprocess_data():
    """Load and preprocess MNIST dataset."""
    print("Loading MNIST dataset...")
    
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    print(f"Original training data shape: {x_train.shape}")
    print(f"Original test data shape: {x_test.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # Normalize pixel values to [0, 1] range
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Flatten 28x28 images to 784-dimensional vectors
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    
    # Convert labels to categorical (one-hot encoding)
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    print(f"Preprocessed training data shape: {x_train.shape}")
    print(f"Preprocessed test data shape: {x_test.shape}")
    print(f"One-hot encoded labels shape: {y_train.shape}")
    
    return (x_train, y_train), (x_test, y_test), num_classes

def visualize_sample_data(x_train, y_train):
    """Visualize sample images from the dataset."""
    print("\nVisualizing sample data...")
    
    # Convert back to original shape for visualization
    x_display = x_train[:16].reshape(-1, 28, 28)
    y_display = np.argmax(y_train[:16], axis=1)
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle('Sample MNIST Images', fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_display[i], cmap='gray')
        ax.set_title(f'Label: {y_display[i]}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_sample_images.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_neural_network(input_shape, num_classes):
    """
    Create a simple feed-forward neural network.
    
    Architecture:
    - Input layer: 784 neurons (28x28 flattened)
    - Hidden layer 1: 128 neurons with ReLU activation
    - Dropout layer: 20% dropout for regularization
    - Hidden layer 2: 64 neurons with ReLU activation
    - Dropout layer: 20% dropout for regularization
    - Output layer: 10 neurons with softmax activation
    """
    print("\nCreating neural network architecture...")
    
    model = keras.Sequential([
        # Input layer (implicitly defined by first Dense layer)
        layers.Dense(128, activation='relu', input_shape=(input_shape,), name='hidden_1'),
        layers.Dropout(0.2, name='dropout_1'),
        
        layers.Dense(64, activation='relu', name='hidden_2'),
        layers.Dropout(0.2, name='dropout_2'),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, x_train, y_train, x_test, y_test):
    """Train the neural network."""
    print("\nTraining the neural network...")
    
    # Define callbacks for better training
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=20,
        batch_size=128,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def plot_training_history(history):
    """Plot training and validation metrics."""
    print("\nVisualizing training history...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    ax1.set_title('Model Accuracy Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss', marker='o')
    ax2.plot(history.history['val_loss'], label='Validation Loss', marker='s')
    ax2.set_title('Model Loss Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mnist_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, x_test, y_test):
    """Evaluate the trained model."""
    print("\nEvaluating the model...")
    
    # Make predictions
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Calculate test accuracy
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Generate classification report
    print("\nClassification Report:")
    class_names = [str(i) for i in range(10)]
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
    
    return y_pred_classes, y_true_classes

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    print("\nGenerating confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix - MNIST Classification')
    plt.xlabel('Predicted Digit')
    plt.ylabel('True Digit')
    plt.tight_layout()
    plt.savefig('mnist_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_predictions(x_test, y_true, y_pred):
    """Visualize some predictions."""
    print("\nVisualizing sample predictions...")
    
    # Convert test data back to image format
    x_images = x_test[:16].reshape(-1, 28, 28)
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('Sample Predictions', fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_images[i], cmap='gray')
        
        # Color code: green for correct, red for incorrect
        color = 'green' if y_true[i] == y_pred[i] else 'red'
        ax.set_title(f'True: {y_true[i]}, Pred: {y_pred[i]}', color=color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_sample_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_model_architecture(model):
    """Analyze and display model architecture details."""
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE ANALYSIS")
    print("="*60)
    
    # Display model summary
    model.summary()
    
    # Calculate total parameters
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    
    # Analyze layer by layer
    print("\nLayer-by-layer analysis:")
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'units'):
            print(f"Layer {i+1} ({layer.name}): {layer.units} units, "
                  f"activation: {layer.activation.__name__}")
        elif hasattr(layer, 'rate'):
            print(f"Layer {i+1} ({layer.name}): dropout rate {layer.rate}")

def main():
    """Main execution function."""
    print("="*70)
    print("SIMPLE FEED-FORWARD NEURAL NETWORK FOR MNIST CLASSIFICATION")
    print("="*70)
    print("Task 3: TensorFlow/Keras Implementation")
    print("="*70)
    
    # Step 1: Load and preprocess data
    print("\nStep 1: Data Loading and Preprocessing")
    print("-" * 40)
    (x_train, y_train), (x_test, y_test), num_classes = load_and_preprocess_data()
    
    # Visualize sample data
    visualize_sample_data(x_train, y_train)
    
    # Step 2: Create neural network architecture
    print("\nStep 2: Neural Network Architecture Design")
    print("-" * 40)
    input_shape = x_train.shape[1]  # 784 features
    model = create_neural_network(input_shape, num_classes)
    
    # Analyze the architecture
    analyze_model_architecture(model)
    
    # Step 3: Train the model
    print("\nStep 3: Model Training")
    print("-" * 40)
    history = train_model(model, x_train, y_train, x_test, y_test)
    
    # Step 4: Visualize training process
    print("\nStep 4: Training Visualization")
    print("-" * 40)
    plot_training_history(history)
    
    # Step 5: Evaluate the model
    print("\nStep 5: Model Evaluation")
    print("-" * 40)
    y_pred, y_true = evaluate_model(model, x_test, y_test)
    
    # Step 6: Visualization of results
    print("\nStep 6: Results Visualization")
    print("-" * 40)
    plot_confusion_matrix(y_true, y_pred)
    visualize_predictions(x_test, y_true, y_pred)
    
    # Step 7: Save the model
    print("\nStep 7: Model Saving")
    print("-" * 40)
    model_filename = 'mnist_neural_network.h5'
    model.save(model_filename)
    print(f"Model saved as: {model_filename}")
    
    # Final summary
    print("\n" + "="*70)
    print("TASK 3 COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("Generated files:")
    print("- mnist_neural_network.h5 (trained model)")
    print("- mnist_sample_images.png (sample dataset images)")
    print("- mnist_training_history.png (training progress)")
    print("- mnist_confusion_matrix.png (classification results)")
    print("- mnist_sample_predictions.png (prediction examples)")
    print("\nThe neural network successfully learned to classify MNIST digits!")
    
    # Display final metrics
    final_accuracy = max(history.history['val_accuracy'])
    print(f"Best validation accuracy achieved: {final_accuracy:.4f}")

if __name__ == "__main__":
    main()
