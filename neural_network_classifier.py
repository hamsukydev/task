"""
Neural Network Classifier with TensorFlow/Keras
===============================================

Interactive web application for neural network classification with customizable
architecture, training visualization, and comprehensive evaluation.

Author: AI Assistant
Date: July 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
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

# Configure Streamlit page
st.set_page_config(
    page_title="Neural Network Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #3498db;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #9b59b6;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitNeuralNetworkClassifier:
    """
    Streamlit Neural Network classifier with interactive interface.
    """
    
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_val = None
        self.y_val = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.X_val_scaled = None
        self.scaler = None
        self.model = None
        self.history = None
        self.feature_names = None
        self.label_encoders = {}
        self.num_classes = None
        
        # Initialize session state
        if 'nn_data_loaded' not in st.session_state:
            st.session_state.nn_data_loaded = False
        if 'nn_model_trained' not in st.session_state:
            st.session_state.nn_model_trained = False
        if 'nn_model' not in st.session_state:
            st.session_state.nn_model = None
        if 'nn_history' not in st.session_state:
            st.session_state.nn_history = None
        if 'nn_feature_names' not in st.session_state:
            st.session_state.nn_feature_names = None
        if 'nn_X_train_scaled' not in st.session_state:
            st.session_state.nn_X_train_scaled = None
        if 'nn_X_test_scaled' not in st.session_state:
            st.session_state.nn_X_test_scaled = None
        if 'nn_X_val_scaled' not in st.session_state:
            st.session_state.nn_X_val_scaled = None
        if 'nn_y_train' not in st.session_state:
            st.session_state.nn_y_train = None
        if 'nn_y_test' not in st.session_state:
            st.session_state.nn_y_test = None
        if 'nn_y_val' not in st.session_state:
            st.session_state.nn_y_val = None
        if 'nn_num_classes' not in st.session_state:
            st.session_state.nn_num_classes = None
        if 'nn_df' not in st.session_state:
            st.session_state.nn_df = None
        if 'nn_preprocessed' not in st.session_state:
            st.session_state.nn_preprocessed = False

    def load_data(self, uploaded_file=None, file_path=None, dataset_type="custom"):
        """Load data from uploaded file, file path, or built-in datasets."""
        try:
            if dataset_type == "mnist":
                # Load MNIST dataset
                (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
                
                # Flatten the images
                X_train = X_train.reshape(X_train.shape[0], -1)
                X_test = X_test.reshape(X_test.shape[0], -1)
                
                # Create DataFrame for consistency
                feature_names = [f'pixel_{i}' for i in range(X_train.shape[1])]
                
                # Combine train and test for consistent processing
                X_combined = np.vstack([X_train, X_test])
                y_combined = np.hstack([y_train, y_test])
                
                self.df = pd.DataFrame(X_combined, columns=feature_names)
                self.df['target'] = y_combined
                
                st.success("✅ MNIST dataset loaded successfully!")
                
            elif dataset_type == "fashion_mnist":
                # Load Fashion-MNIST dataset
                (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
                
                # Flatten the images
                X_train = X_train.reshape(X_train.shape[0], -1)
                X_test = X_test.reshape(X_test.shape[0], -1)
                
                # Create DataFrame for consistency
                feature_names = [f'pixel_{i}' for i in range(X_train.shape[1])]
                
                # Combine train and test for consistent processing
                X_combined = np.vstack([X_train, X_test])
                y_combined = np.hstack([y_train, y_test])
                
                self.df = pd.DataFrame(X_combined, columns=feature_names)
                self.df['target'] = y_combined
                
                st.success("✅ Fashion-MNIST dataset loaded successfully!")
                
            elif uploaded_file is not None:
                self.df = pd.read_csv(uploaded_file)
                st.success("✅ Custom dataset loaded successfully!")
                
            elif file_path is not None:
                self.df = pd.read_csv(file_path)
                st.success("✅ Default dataset loaded successfully!")
                
            else:
                return False
            
            # Store dataframe in session state
            st.session_state.nn_df = self.df
            st.session_state.nn_data_loaded = True
            return True
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False

    def display_data_overview(self):
        """Display data overview and statistics."""
        # Load dataframe from session state if not available
        if self.df is None and st.session_state.nn_df is not None:
            self.df = st.session_state.nn_df
        
        if self.df is None:
            st.warning("No data available for overview.")
            return
            
        st.markdown('<h2 class="sub-header">📊 Data Overview</h2>', unsafe_allow_html=True)
        
        # Determine target column
        target_col = 'target' if 'target' in self.df.columns else 'Churn'
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", self.df.shape[0])
        with col2:
            st.metric("Total Features", self.df.shape[1] - 1)
        with col3:
            st.metric("Missing Values", self.df.isnull().sum().sum())
        with col4:
            if target_col in self.df.columns:
                num_classes = self.df[target_col].nunique()
                st.metric("Number of Classes", num_classes)
        
        # Display first few rows (sample for large datasets)
        st.subheader("Data Sample")
        if self.df.shape[1] > 50:  # For image datasets
            st.info("Showing sample of features due to large dimensionality")
            sample_cols = [target_col] + list(self.df.columns[:10]) + ['...']
            sample_df = self.df[[target_col] + list(self.df.columns[:10])].head()
            sample_df['...'] = '...'
            st.dataframe(sample_df)
        else:
            st.dataframe(self.df.head())
        
        # Basic statistics
        if self.df.shape[1] <= 20:  # Only show for smaller datasets
            st.subheader("Basic Statistics")
            st.dataframe(self.df.describe())
        
        # Target distribution
        if target_col in self.df.columns:
            st.subheader("Target Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                target_counts = self.df[target_col].value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(10, 6))
                target_counts.plot(kind='bar', ax=ax, color='skyblue')
                ax.set_title("Class Distribution")
                ax.set_xlabel("Class")
                ax.set_ylabel("Count")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                target_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
                ax.set_title("Class Distribution (Percentage)")
                ax.set_ylabel("")
                st.pyplot(fig)
        
        # Show sample images for image datasets
        if 'pixel_' in str(self.df.columns[0]) and target_col in self.df.columns:
            st.subheader("Sample Images")
            self._display_sample_images(target_col)

    def _display_sample_images(self, target_col):
        """Display sample images for image datasets."""
        try:
            # Calculate image dimensions
            n_features = self.df.shape[1] - 1
            img_size = int(np.sqrt(n_features))
            
            if img_size * img_size == n_features:
                # Display samples from each class
                classes = sorted(self.df[target_col].unique())
                n_classes = min(len(classes), 10)  # Show max 10 classes
                
                fig, axes = plt.subplots(2, n_classes, figsize=(2*n_classes, 4))
                if n_classes == 1:
                    axes = axes.reshape(2, 1)
                
                for i, class_label in enumerate(classes[:n_classes]):
                    class_samples = self.df[self.df[target_col] == class_label]
                    
                    # Show 2 samples per class
                    for j in range(min(2, len(class_samples))):
                        sample = class_samples.iloc[j, :-1].values.reshape(img_size, img_size)
                        axes[j, i].imshow(sample, cmap='gray')
                        axes[j, i].set_title(f'Class {class_label}')
                        axes[j, i].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
                
        except Exception as e:
            st.warning(f"Could not display sample images: {str(e)}")

    def preprocess_data(self):
        """Preprocess the data for neural network training."""
        # Load dataframe from session state if not available
        if self.df is None and st.session_state.nn_df is not None:
            self.df = st.session_state.nn_df
        
        if self.df is None:
            st.error("No data available for preprocessing. Please load data first.")
            return
            
        with st.spinner("Preprocessing data..."):
            # Determine target column
            target_col = 'target' if 'target' in self.df.columns else 'Churn'
            
            if target_col not in self.df.columns:
                st.error("No target column found. Please ensure your dataset has a 'target' or 'Churn' column.")
                return
            
            # Make a copy of the dataframe
            df_processed = self.df.copy()
            
            # Handle target variable
            y = df_processed[target_col]
            
            # Convert target to numeric if needed
            if y.dtype == 'object' or y.dtype == 'bool':
                if target_col == 'Churn':
                    # Handle churn dataset
                    churn_map = {'True': 1, 'False': 0, True: 1, False: 0}
                    y = y.map(churn_map)
                else:
                    # Use label encoder for other categorical targets
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    self.label_encoders[target_col] = le
            
            # Encode other categorical variables
            X = df_processed.drop(target_col, axis=1)
            categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
            
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                self.label_encoders[col] = le
            
            self.feature_names = X.columns.tolist()
            self.num_classes = len(np.unique(y))
            
            # Store in session state
            st.session_state.nn_feature_names = self.feature_names
            st.session_state.nn_num_classes = self.num_classes
            
            # Split the data
            test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05, key="nn_test_size")
            val_size = st.sidebar.slider("Validation Size", 0.1, 0.3, 0.2, 0.05, key="nn_val_size")
            
            # First split: separate test set
            X_temp, self.X_test, y_temp, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Second split: separate train and validation
            val_size_adjusted = val_size / (1 - test_size)
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
            )
            
            # Scale the features
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            self.X_val_scaled = self.scaler.transform(self.X_val)
            
            # Store in session state
            st.session_state.nn_X_train_scaled = self.X_train_scaled
            st.session_state.nn_X_test_scaled = self.X_test_scaled
            st.session_state.nn_X_val_scaled = self.X_val_scaled
            st.session_state.nn_y_train = self.y_train
            st.session_state.nn_y_test = self.y_test
            st.session_state.nn_y_val = self.y_val
            st.session_state.nn_preprocessed = True
            
            st.success(f"✅ Data preprocessed successfully!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.info(f"Training set: {self.X_train_scaled.shape[0]} samples")
            with col2:
                st.info(f"Validation set: {self.X_val_scaled.shape[0]} samples")
            with col3:
                st.info(f"Test set: {self.X_test_scaled.shape[0]} samples")
            with col4:
                st.info(f"Features: {self.X_train_scaled.shape[1]}")

    def design_neural_network(self):
        """Design the neural network architecture."""
        
        # Load data from session state if available
        if st.session_state.nn_X_train_scaled is not None:
            self.X_train_scaled = st.session_state.nn_X_train_scaled
            self.y_train = st.session_state.nn_y_train
            self.X_val_scaled = st.session_state.nn_X_val_scaled
            self.y_val = st.session_state.nn_y_val
            self.feature_names = st.session_state.nn_feature_names
            self.num_classes = st.session_state.nn_num_classes
        
        if self.X_train_scaled is None:
            st.warning("Please preprocess the data first!")
            return
        
        st.markdown('<h3 class="sub-header">🧠 Neural Network Architecture</h3>', unsafe_allow_html=True)
        
        # Architecture parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Hidden Layers Configuration")
            num_hidden_layers = st.slider("Number of Hidden Layers", 1, 5, 2, key="nn_hidden_layers")
            
            hidden_layers = []
            for i in range(num_hidden_layers):
                neurons = st.slider(f"Layer {i+1} Neurons", 16, 512, 128, 16, key=f"nn_layer_{i}")
                hidden_layers.append(neurons)
            
            activation = st.selectbox("Hidden Layer Activation", 
                                    ['relu', 'tanh', 'sigmoid'], key="nn_activation")
            dropout_rate = st.slider("Dropout Rate", 0.0, 0.8, 0.2, 0.1, key="nn_dropout")
        
        with col2:
            st.subheader("Training Configuration")
            optimizer = st.selectbox("Optimizer", 
                                    ['adam', 'rmsprop', 'sgd'], key="nn_optimizer")
            learning_rate = st.selectbox("Learning Rate", 
                                       [0.001, 0.01, 0.1], key="nn_lr")
            batch_size = st.selectbox("Batch Size", 
                                    [16, 32, 64, 128], key="nn_batch_size")
            epochs = st.slider("Epochs", 10, 200, 50, 10, key="nn_epochs")
        
        # Display architecture summary
        st.subheader("Architecture Summary")
        
        input_dim = self.X_train_scaled.shape[1]
        
        architecture_info = []
        architecture_info.append(f"Input Layer: {input_dim} neurons")
        
        for i, neurons in enumerate(hidden_layers):
            architecture_info.append(f"Hidden Layer {i+1}: {neurons} neurons ({activation})")
        
        if self.num_classes > 2:
            output_activation = "softmax"
            loss_function = "sparse_categorical_crossentropy"
        else:
            output_activation = "sigmoid"
            loss_function = "binary_crossentropy"
        
        architecture_info.append(f"Output Layer: {self.num_classes} neurons ({output_activation})")
        architecture_info.append(f"Loss Function: {loss_function}")
        architecture_info.append(f"Optimizer: {optimizer} (lr={learning_rate})")
        
        for info in architecture_info:
            st.write(f"• {info}")
        
        # Training form
        with st.form("nn_training_form"):
            submitted = st.form_submit_button("🚀 Build and Train Neural Network")
            
            if submitted:
                with st.spinner("Building and training neural network..."):
                    # Build model
                    model = self._build_model(
                        input_dim, hidden_layers, self.num_classes,
                        activation, dropout_rate, output_activation,
                        optimizer, learning_rate, loss_function
                    )
                    
                    # Display model summary
                    st.subheader("Model Summary")
                    
                    # Capture model summary
                    summary_list = []
                    model.summary(print_fn=lambda x: summary_list.append(x))
                    summary_text = '\n'.join(summary_list)
                    st.text(summary_text)
                    
                    # Train model
                    start_time = time.time()
                    
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Custom callback for progress updates
                    class StreamlitCallback(keras.callbacks.Callback):
                        def __init__(self, progress_bar, status_text, total_epochs):
                            self.progress_bar = progress_bar
                            self.status_text = status_text
                            self.total_epochs = total_epochs
                        
                        def on_epoch_end(self, epoch, logs=None):
                            progress = (epoch + 1) / self.total_epochs
                            self.progress_bar.progress(progress)
                            
                            loss = logs.get('loss', 0)
                            val_loss = logs.get('val_loss', 0)
                            accuracy = logs.get('accuracy', 0)
                            val_accuracy = logs.get('val_accuracy', 0)
                            
                            self.status_text.text(
                                f"Epoch {epoch + 1}/{self.total_epochs} - "
                                f"Loss: {loss:.4f} - Val Loss: {val_loss:.4f} - "
                                f"Accuracy: {accuracy:.4f} - Val Accuracy: {val_accuracy:.4f}"
                            )
                    
                    # Train the model
                    history = model.fit(
                        self.X_train_scaled, self.y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(self.X_val_scaled, self.y_val),
                        callbacks=[StreamlitCallback(progress_bar, status_text, epochs)],
                        verbose=0
                    )
                    
                    training_time = time.time() - start_time
                    
                    # Store results
                    self.model = model
                    self.history = history
                    st.session_state.nn_model = model
                    st.session_state.nn_history = history.history
                    st.session_state.nn_model_trained = True
                    
                    progress_bar.progress(1.0)
                    status_text.text("Training completed!")
                    
                    st.success(f"✅ Model trained successfully in {training_time:.2f} seconds!")

    def _build_model(self, input_dim, hidden_layers, num_classes, 
                     activation, dropout_rate, output_activation,
                     optimizer, learning_rate, loss_function):
        """Build the neural network model."""
        
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(hidden_layers[0], 
                              activation=activation, 
                              input_dim=input_dim,
                              name='input_layer'))
        model.add(layers.Dropout(dropout_rate))
        
        # Hidden layers
        for i, neurons in enumerate(hidden_layers[1:], 1):
            model.add(layers.Dense(neurons, 
                                  activation=activation,
                                  name=f'hidden_layer_{i}'))
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(layers.Dense(num_classes if num_classes > 2 else 1, 
                              activation=output_activation,
                              name='output_layer'))
        
        # Compile model
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = keras.optimizers.SGD(learning_rate=learning_rate)
        
        model.compile(
            optimizer=opt,
            loss=loss_function,
            metrics=['accuracy']
        )
        
        return model

    def evaluate_model(self):
        """Evaluate the trained neural network."""
        # Load model from session state
        if st.session_state.nn_model is not None:
            self.model = st.session_state.nn_model
            self.history = type('History', (), {'history': st.session_state.nn_history})()
        
        if not st.session_state.nn_model_trained or self.model is None:
            st.warning("Please train a neural network model first!")
            return
        
        # Load data from session state
        if st.session_state.nn_X_test_scaled is not None:
            self.X_test_scaled = st.session_state.nn_X_test_scaled
            self.y_test = st.session_state.nn_y_test
            self.num_classes = st.session_state.nn_num_classes
        
        st.markdown('<h2 class="sub-header">📈 Model Evaluation</h2>', unsafe_allow_html=True)
        
        # Make predictions
        y_pred_proba = self.model.predict(self.X_test_scaled)
        
        if self.num_classes > 2:
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        
        if self.num_classes == 2:
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
        else:
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        # Display metrics
        st.subheader("Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
        with col2:
            st.metric("Precision", f"{precision:.4f}")
        with col3:
            st.metric("Recall", f"{recall:.4f}")
        with col4:
            st.metric("F1-Score", f"{f1:.4f}")
        
        # Visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Training History", "📊 Confusion Matrix", "🎯 Predictions", "📋 Classification Report"])
        
        with tab1:
            self._plot_training_history()
        
        with tab2:
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
        
        with tab3:
            # Prediction analysis
            self._analyze_predictions(y_pred, y_pred_proba)
        
        with tab4:
            # Classification Report
            if self.num_classes > 2:
                target_names = [f'Class {i}' for i in range(self.num_classes)]
            else:
                target_names = ['Class 0', 'Class 1']
            
            report = classification_report(self.y_test, y_pred, 
                                         target_names=target_names, 
                                         output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

    def _plot_training_history(self):
        """Plot training and validation history."""
        if self.history is None:
            st.warning("No training history available.")
            return
        
        history_dict = self.history.history
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        epochs = range(1, len(history_dict['loss']) + 1)
        ax1.plot(epochs, history_dict['loss'], 'bo-', label='Training Loss')
        ax1.plot(epochs, history_dict['val_loss'], 'ro-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(epochs, history_dict['accuracy'], 'bo-', label='Training Accuracy')
        ax2.plot(epochs, history_dict['val_accuracy'], 'ro-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Training summary
        st.subheader("Training Summary")
        final_train_loss = history_dict['loss'][-1]
        final_val_loss = history_dict['val_loss'][-1]
        final_train_acc = history_dict['accuracy'][-1]
        final_val_acc = history_dict['val_accuracy'][-1]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Final Training Loss", f"{final_train_loss:.4f}")
        with col2:
            st.metric("Final Validation Loss", f"{final_val_loss:.4f}")
        with col3:
            st.metric("Final Training Accuracy", f"{final_train_acc:.4f}")
        with col4:
            st.metric("Final Validation Accuracy", f"{final_val_acc:.4f}")

    def _analyze_predictions(self, y_pred, y_pred_proba):
        """Analyze model predictions."""
        st.subheader("Prediction Analysis")
        
        # Prediction confidence distribution
        if self.num_classes == 2:
            confidence = np.abs(y_pred_proba.flatten() - 0.5) + 0.5
        else:
            confidence = np.max(y_pred_proba, axis=1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Confidence distribution
        ax1.hist(confidence, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Prediction Confidence Distribution')
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Correct vs Incorrect predictions by confidence
        correct_mask = (y_pred == self.y_test)
        
        ax2.hist(confidence[correct_mask], bins=20, alpha=0.7, 
                label='Correct', color='green', edgecolor='black')
        ax2.hist(confidence[~correct_mask], bins=20, alpha=0.7, 
                label='Incorrect', color='red', edgecolor='black')
        ax2.set_title('Prediction Accuracy by Confidence')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show some example predictions
        st.subheader("Sample Predictions")
        
        # Get indices for high and low confidence predictions
        high_conf_idx = np.argsort(confidence)[-5:]
        low_conf_idx = np.argsort(confidence)[:5]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Highest Confidence Predictions:**")
            for idx in high_conf_idx:
                actual = self.y_test.iloc[idx] if hasattr(self.y_test, 'iloc') else self.y_test[idx]
                predicted = y_pred[idx]
                conf = confidence[idx]
                status = "✅" if actual == predicted else "❌"
                st.write(f"{status} Actual: {actual}, Predicted: {predicted}, Confidence: {conf:.3f}")
        
        with col2:
            st.write("**Lowest Confidence Predictions:**")
            for idx in low_conf_idx:
                actual = self.y_test.iloc[idx] if hasattr(self.y_test, 'iloc') else self.y_test[idx]
                predicted = y_pred[idx]
                conf = confidence[idx]
                status = "✅" if actual == predicted else "❌"
                st.write(f"{status} Actual: {actual}, Predicted: {predicted}, Confidence: {conf:.3f}")

def main():
    """Main Streamlit application."""
    
    # Title and description
    st.markdown('<h1 class="main-header">🧠 Neural Network Classifier</h1>', unsafe_allow_html=True)
    st.markdown("**Interactive Deep Learning Application with TensorFlow/Keras**")
    
    # Initialize classifier
    classifier = StreamlitNeuralNetworkClassifier()
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Control Panel")
    
    # Reset button
    if st.sidebar.button("🔄 Reset All", key="nn_reset"):
        for key in list(st.session_state.keys()):
            if key.startswith('nn_'):
                del st.session_state[key]
        st.rerun()
    
    # Data loading options
    st.sidebar.markdown("### 📁 Data Loading")
    data_option = st.sidebar.radio(
        "Choose data source:",
        ["MNIST Digits", "Fashion-MNIST", "Upload CSV File", "Use Default Churn Dataset"],
        key="nn_data_option"
    )
    
    if data_option == "Upload CSV File":
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv", key="nn_upload")
        if uploaded_file is not None:
            if classifier.load_data(uploaded_file=uploaded_file, dataset_type="custom"):
                st.sidebar.success("✅ Data loaded successfully!")
    
    elif data_option == "Use Default Churn Dataset":
        default_path = r"c:\Users\ITAPPS002\OneDrive\Documents\Data Set For Task\Churn Prdiction Data\churn-bigml-80.csv"
        if st.sidebar.button("Load Default Dataset", key="nn_load_default"):
            if classifier.load_data(file_path=default_path, dataset_type="custom"):
                st.sidebar.success("✅ Default dataset loaded!")
    
    elif data_option == "MNIST Digits":
        if st.sidebar.button("Load MNIST Dataset", key="nn_load_mnist"):
            if classifier.load_data(dataset_type="mnist"):
                st.sidebar.success("✅ MNIST dataset loaded!")
    
    elif data_option == "Fashion-MNIST":
        if st.sidebar.button("Load Fashion-MNIST Dataset", key="nn_load_fashion"):
            if classifier.load_data(dataset_type="fashion_mnist"):
                st.sidebar.success("✅ Fashion-MNIST dataset loaded!")
    
    # Main content
    if st.session_state.nn_data_loaded:
        # Load dataframe from session state if needed
        if classifier.df is None and st.session_state.nn_df is not None:
            classifier.df = st.session_state.nn_df
        
        if classifier.df is not None:
            # Data overview
            classifier.display_data_overview()
            
            # Preprocessing
            st.markdown('<h2 class="sub-header">🔧 Data Preprocessing</h2>', unsafe_allow_html=True)
            
            if not st.session_state.nn_preprocessed:
                if st.button("Preprocess Data", key="nn_preprocess"):
                    classifier.preprocess_data()
            else:
                st.success("✅ Data has been preprocessed!")
                # Show preprocessing summary
                if st.session_state.nn_X_train_scaled is not None:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.info(f"Training set: {st.session_state.nn_X_train_scaled.shape[0]} samples")
                    with col2:
                        st.info(f"Validation set: {st.session_state.nn_X_val_scaled.shape[0]} samples")
                    with col3:
                        st.info(f"Test set: {st.session_state.nn_X_test_scaled.shape[0]} samples")
                    with col4:
                        st.info(f"Features: {st.session_state.nn_X_train_scaled.shape[1]}")
            
            # Model design and training
            if (st.session_state.nn_preprocessed and 
                st.session_state.nn_X_train_scaled is not None):
                classifier.design_neural_network()
            
            # Model evaluation
            if st.session_state.nn_model_trained:
                classifier.evaluate_model()
    
    else:
        st.info("👆 Please load a dataset to get started!")
        
        # Show information about available datasets
        st.markdown("### 📝 Available Datasets")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Built-in Datasets:**
            - **MNIST Digits**: 70,000 handwritten digits (0-9)
            - **Fashion-MNIST**: 70,000 fashion items (10 categories)
            
            **Custom Datasets:**
            - Upload your own CSV file
            - Use the default churn prediction dataset
            """)
        
        with col2:
            st.markdown("""
            **Neural Network Features:**
            - Customizable architecture (layers, neurons, activation)
            - Multiple optimizers (Adam, RMSprop, SGD)
            - Real-time training visualization
            - Comprehensive evaluation metrics
            - Prediction confidence analysis
            """)

if __name__ == "__main__":
    main()
