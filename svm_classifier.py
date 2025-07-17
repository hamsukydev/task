"""
Support Vector Machine (SVM) Classifier for Binary Classification
================================================================

Interactive web application for SVM classification with different kernels,
decision boundary visualization, and comprehensive performance evaluation.

Author: HamsukyTech
Date: July 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    StratifiedKFold, validation_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve
)
from sklearn.decomposition import PCA
import warnings
import time
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="SVM Classifier",
    page_icon="⚡",
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
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #8e44ad;
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

class StreamlitSVMClassifier:
    """
    Streamlit SVM classifier with interactive interface.
    """
    
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.scaler = None
        self.models = {}
        self.feature_names = None
        self.label_encoders = {}
        
        # Initialize session state
        if 'svm_data_loaded' not in st.session_state:
            st.session_state.svm_data_loaded = False
        if 'svm_models_trained' not in st.session_state:
            st.session_state.svm_models_trained = False
        if 'svm_models' not in st.session_state:
            st.session_state.svm_models = {}
        if 'svm_feature_names' not in st.session_state:
            st.session_state.svm_feature_names = None
        if 'svm_X_train' not in st.session_state:
            st.session_state.svm_X_train = None
        if 'svm_X_test' not in st.session_state:
            st.session_state.svm_X_test = None
        if 'svm_y_train' not in st.session_state:
            st.session_state.svm_y_train = None
        if 'svm_y_test' not in st.session_state:
            st.session_state.svm_y_test = None
        if 'svm_X_train_scaled' not in st.session_state:
            st.session_state.svm_X_train_scaled = None
        if 'svm_X_test_scaled' not in st.session_state:
            st.session_state.svm_X_test_scaled = None
        if 'svm_scaler' not in st.session_state:
            st.session_state.svm_scaler = None

    def load_data(self, uploaded_file=None, file_path=None):
        """Load data from uploaded file or file path."""
        try:
            if uploaded_file is not None:
                self.df = pd.read_csv(uploaded_file)
            elif file_path is not None:
                self.df = pd.read_csv(file_path)
            else:
                return False
            
            st.session_state.svm_data_loaded = True
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False

    def display_data_overview(self):
        """Display data overview and statistics."""
        st.markdown('<h2 class="sub-header">📊 Data Overview</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", self.df.shape[0])
        with col2:
            st.metric("Total Columns", self.df.shape[1])
        with col3:
            st.metric("Missing Values", self.df.isnull().sum().sum())
        with col4:
            churn_rate = (self.df['Churn'].sum() / len(self.df) * 100) if 'Churn' in self.df.columns else 0
            st.metric("Positive Class Rate", f"{churn_rate:.1f}%")
        
        # Display first few rows
        st.subheader("First 5 Rows")
        st.dataframe(self.df.head())
        
        # Basic statistics
        st.subheader("Basic Statistics")
        st.dataframe(self.df.describe())
        
        # Target distribution
        if 'Churn' in self.df.columns:
            st.subheader("Target Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                churn_counts = self.df['Churn'].value_counts()
                fig, ax = plt.subplots(figsize=(8, 6))
                churn_counts.plot(kind='bar', ax=ax, color=['skyblue', 'lightcoral'])
                ax.set_title("Class Distribution")
                ax.set_xlabel("Churn")
                ax.set_ylabel("Count")
                plt.xticks(rotation=0)
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                churn_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=['skyblue', 'lightcoral'])
                ax.set_title("Class Distribution (Percentage)")
                ax.set_ylabel("")
                st.pyplot(fig)

    def preprocess_data(self):
        """Preprocess the data for machine learning."""
        with st.spinner("Preprocessing data..."):
            # Make a copy of the dataframe
            df_processed = self.df.copy()
            
            # Convert boolean target to numeric
            if df_processed['Churn'].dtype == 'bool':
                df_processed['Churn'] = df_processed['Churn'].astype(int)
            elif df_processed['Churn'].dtype == 'object':
                # Handle string representations
                churn_map = {'True': 1, 'False': 0, True: 1, False: 0}
                df_processed['Churn'] = df_processed['Churn'].map(churn_map)
            
            # Encode categorical variables
            categorical_columns = df_processed.select_dtypes(include=['object']).columns.tolist()
            if 'Churn' in categorical_columns:
                categorical_columns.remove('Churn')
            
            for col in categorical_columns:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                self.label_encoders[col] = le
            
            # Separate features and target
            X = df_processed.drop('Churn', axis=1)
            y = df_processed['Churn']
            
            self.feature_names = X.columns.tolist()
            st.session_state.svm_feature_names = self.feature_names
            
            # Split the data
            test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05, key="svm_test_size")
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Scale the features (important for SVM)
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            # Store in session state
            st.session_state.svm_X_train = self.X_train
            st.session_state.svm_X_test = self.X_test
            st.session_state.svm_y_train = self.y_train
            st.session_state.svm_y_test = self.y_test
            st.session_state.svm_X_train_scaled = self.X_train_scaled
            st.session_state.svm_X_test_scaled = self.X_test_scaled
            st.session_state.svm_scaler = self.scaler
            
            st.success(f"✅ Data preprocessed successfully!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"Training set: {self.X_train.shape[0]} samples")
            with col2:
                st.info(f"Test set: {self.X_test.shape[0]} samples")
            with col3:
                st.info(f"Features: {self.X_train.shape[1]}")

    def train_svm_models(self):
        """Train SVM models with different kernels."""
        
        # Load data from session state if available
        if (st.session_state.svm_X_train_scaled is not None and 
            st.session_state.svm_y_train is not None):
            self.X_train_scaled = st.session_state.svm_X_train_scaled
            self.X_test_scaled = st.session_state.svm_X_test_scaled
            self.y_train = st.session_state.svm_y_train
            self.y_test = st.session_state.svm_y_test
            self.feature_names = st.session_state.svm_feature_names
            self.scaler = st.session_state.svm_scaler
        
        if self.X_train_scaled is None:
            st.warning("Please preprocess the data first!")
            return
        
        st.markdown('<h3 class="sub-header">⚡ SVM Model Training</h3>', unsafe_allow_html=True)
        
        # Kernel selection
        st.subheader("Kernel Selection")
        col1, col2 = st.columns(2)
        
        with col1:
            kernels = st.multiselect(
                "Select kernels to train:",
                ['linear', 'rbf', 'poly', 'sigmoid'],
                default=['linear', 'rbf'],
                key="svm_kernels"
            )
        
        with col2:
            use_grid_search = st.checkbox("Use Grid Search for hyperparameter tuning", value=True)
        
        # Training form
        with st.form("svm_training_form"):
            submitted = st.form_submit_button("🚀 Train SVM Models")
            
            if submitted:
                if not kernels:
                    st.error("Please select at least one kernel!")
                    return
                
                self.models = {}
                
                for kernel in kernels:
                    st.write(f"Training SVM with {kernel} kernel...")
                    
                    with st.spinner(f"Training {kernel} kernel..."):
                        if use_grid_search:
                            # Define parameter grid based on kernel
                            if kernel == 'linear':
                                param_grid = {'C': [0.1, 1, 10, 100]}
                            elif kernel == 'rbf':
                                param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]}
                            elif kernel == 'poly':
                                param_grid = {'C': [0.1, 1, 10], 'degree': [2, 3, 4], 'gamma': ['scale', 'auto']}
                            else:  # sigmoid
                                param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
                            
                            svm = SVC(kernel=kernel, probability=True, random_state=42)
                            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                            
                            grid_search = GridSearchCV(
                                svm, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=0
                            )
                            
                            start_time = time.time()
                            grid_search.fit(self.X_train_scaled, self.y_train)
                            training_time = time.time() - start_time
                            
                            self.models[kernel] = {
                                'model': grid_search.best_estimator_,
                                'best_params': grid_search.best_params_,
                                'best_score': grid_search.best_score_,
                                'training_time': training_time
                            }
                            
                        else:
                            # Use default parameters
                            svm = SVC(kernel=kernel, probability=True, random_state=42)
                            
                            start_time = time.time()
                            svm.fit(self.X_train_scaled, self.y_train)
                            training_time = time.time() - start_time
                            
                            self.models[kernel] = {
                                'model': svm,
                                'best_params': 'Default',
                                'best_score': 'N/A',
                                'training_time': training_time
                            }
                
                st.session_state.svm_models = self.models
                st.session_state.svm_models_trained = True
                
                st.success(f"✅ Successfully trained {len(kernels)} SVM models!")
                
                # Display training summary
                st.subheader("Training Summary")
                summary_data = []
                for kernel, model_info in self.models.items():
                    summary_data.append({
                        'Kernel': kernel,
                        'Best Parameters': str(model_info['best_params']),
                        'CV F1-Score': f"{model_info['best_score']:.4f}" if model_info['best_score'] != 'N/A' else 'N/A',
                        'Training Time (s)': f"{model_info['training_time']:.2f}"
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)

    def evaluate_models(self):
        """Evaluate all trained SVM models."""
        # Load models from session state
        if st.session_state.svm_models:
            self.models = st.session_state.svm_models
        
        if not st.session_state.svm_models_trained or not self.models:
            st.warning("Please train SVM models first!")
            return
        
        # Load data from session state
        if (st.session_state.svm_X_test_scaled is not None and 
            st.session_state.svm_y_test is not None):
            self.X_test_scaled = st.session_state.svm_X_test_scaled
            self.y_test = st.session_state.svm_y_test
        
        st.markdown('<h2 class="sub-header">📈 Model Evaluation</h2>', unsafe_allow_html=True)
        
        # Calculate metrics for all models
        all_metrics = {}
        
        for kernel, model_info in self.models.items():
            model = model_info['model']
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            metrics = {
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred),
                'Recall': recall_score(self.y_test, y_pred),
                'F1-Score': f1_score(self.y_test, y_pred),
                'AUC': roc_auc_score(self.y_test, y_pred_proba)
            }
            
            all_metrics[kernel] = metrics
        
        # Display metrics comparison
        st.subheader("Performance Comparison")
        metrics_df = pd.DataFrame(all_metrics).T
        st.dataframe(metrics_df.round(4), use_container_width=True)
        
        # Visualize metrics comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        metrics_df.plot(kind='bar', ax=ax)
        ax.set_title('SVM Models Performance Comparison')
        ax.set_ylabel('Score')
        ax.set_xlabel('Kernel')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Detailed evaluation for each model
        for kernel, model_info in self.models.items():
            with st.expander(f"📊 Detailed Analysis - {kernel.upper()} Kernel"):
                self._detailed_model_evaluation(kernel, model_info)
        
        return all_metrics

    def _detailed_model_evaluation(self, kernel, model_info):
        """Detailed evaluation for a single model."""
        model = model_info['model']
        y_pred = model.predict(self.X_test_scaled)
        y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Metrics
        metrics = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred),
            'Recall': recall_score(self.y_test, y_pred),
            'F1-Score': f1_score(self.y_test, y_pred),
            'AUC': roc_auc_score(self.y_test, y_pred_proba)
        }
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        cols = [col1, col2, col3, col4, col5]
        for i, (metric, value) in enumerate(metrics.items()):
            with cols[i]:
                st.metric(metric, f"{value:.4f}")
        
        # Visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Confusion Matrix", "📈 ROC Curve", "🎯 Precision-Recall", "📋 Classification Report"])
        
        with tab1:
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'Confusion Matrix - {kernel.upper()} Kernel')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
        
        with tab2:
            # ROC Curve
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f'{kernel.upper()} (AUC = {metrics["AUC"]:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - {kernel.upper()} Kernel')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        
        with tab3:
            # Precision-Recall Curve
            precision_curve, recall_curve, _ = precision_recall_curve(self.y_test, y_pred_proba)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(recall_curve, precision_curve)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'Precision-Recall Curve - {kernel.upper()} Kernel')
            ax.grid(True)
            st.pyplot(fig)
        
        with tab4:
            # Classification Report
            report = classification_report(self.y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

    def visualize_decision_boundary(self):
        """Visualize decision boundaries for SVM models."""
        # Load models and data from session state
        if st.session_state.svm_models:
            self.models = st.session_state.svm_models
        
        if not st.session_state.svm_models_trained or not self.models:
            st.warning("Please train SVM models first!")
            return
        
        # Load data from session state
        if (st.session_state.svm_X_train_scaled is not None and 
            st.session_state.svm_y_train is not None):
            self.X_train_scaled = st.session_state.svm_X_train_scaled
            self.y_train = st.session_state.svm_y_train
            self.feature_names = st.session_state.svm_feature_names
        
        st.markdown('<h2 class="sub-header">🎯 Decision Boundary Visualization</h2>', unsafe_allow_html=True)
        
        st.info("For visualization purposes, we'll use PCA to reduce the data to 2D")
        
        # Reduce to 2D using PCA
        pca = PCA(n_components=2)
        X_train_2d = pca.fit_transform(self.X_train_scaled)
        
        # Show PCA explained variance
        explained_variance = pca.explained_variance_ratio_
        st.write(f"PCA Explained Variance: {explained_variance[0]:.3f} + {explained_variance[1]:.3f} = {explained_variance.sum():.3f}")
        
        # Select kernel for visualization
        available_kernels = list(self.models.keys())
        selected_kernel = st.selectbox("Select kernel for decision boundary visualization:", available_kernels)
        
        if st.button("🎨 Generate Decision Boundary"):
            with st.spinner("Generating decision boundary visualization..."):
                # Train a new SVM on 2D data for visualization
                model_2d = SVC(kernel=selected_kernel, probability=True, random_state=42)
                model_2d.fit(X_train_2d, self.y_train)
                
                # Create a mesh
                h = 0.02  # step size in the mesh
                x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
                y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                   np.arange(y_min, y_max, h))
                
                # Plot the decision boundary
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Decision boundary plot
                Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                ax1.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
                scatter = ax1.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=self.y_train, 
                                    cmap=plt.cm.RdYlBu, edgecolors='black')
                ax1.set_xlabel('First Principal Component')
                ax1.set_ylabel('Second Principal Component')
                ax1.set_title(f'SVM Decision Boundary - {selected_kernel.upper()} Kernel')
                plt.colorbar(scatter, ax=ax1)
                
                # Support vectors
                if hasattr(model_2d, 'support_vectors_'):
                    ax1.scatter(model_2d.support_vectors_[:, 0], model_2d.support_vectors_[:, 1],
                              s=100, facecolors='none', edgecolors='red', linewidth=2,
                              label=f'Support Vectors ({len(model_2d.support_vectors_)})')
                    ax1.legend()
                
                # Probability contours
                Z_proba = model_2d.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
                Z_proba = Z_proba.reshape(xx.shape)
                
                contour = ax2.contourf(xx, yy, Z_proba, levels=20, alpha=0.6, cmap=plt.cm.RdYlBu)
                ax2.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=self.y_train, 
                          cmap=plt.cm.RdYlBu, edgecolors='black')
                ax2.set_xlabel('First Principal Component')
                ax2.set_ylabel('Second Principal Component')
                ax2.set_title(f'Class Probability - {selected_kernel.upper()} Kernel')
                plt.colorbar(contour, ax=ax2)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display model information
                st.subheader("Model Information")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if hasattr(model_2d, 'support_vectors_'):
                        st.metric("Support Vectors", len(model_2d.support_vectors_))
                
                with col2:
                    if hasattr(model_2d, 'n_support_'):
                        st.metric("Support Vectors per Class", f"{model_2d.n_support_[0]}, {model_2d.n_support_[1]}")
                
                with col3:
                    if hasattr(model_2d, 'intercept_'):
                        st.metric("Intercept", f"{model_2d.intercept_[0]:.4f}")

    def cross_validation_analysis(self):
        """Perform cross-validation analysis for SVM models."""
        # Load models from session state
        if st.session_state.svm_models:
            self.models = st.session_state.svm_models
        
        if not st.session_state.svm_models_trained or not self.models:
            st.warning("Please train SVM models first!")
            return
        
        # Load data from session state
        if (st.session_state.svm_X_train_scaled is not None and 
            st.session_state.svm_y_train is not None):
            self.X_train_scaled = st.session_state.svm_X_train_scaled
            self.y_train = st.session_state.svm_y_train
        
        st.markdown('<h2 class="sub-header">🔄 Cross-Validation Analysis</h2>', unsafe_allow_html=True)
        
        with st.spinner("Performing cross-validation..."):
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            
            cv_results = {}
            
            for kernel, model_info in self.models.items():
                model = model_info['model']
                kernel_results = {}
                
                for metric in scoring_metrics:
                    scores = cross_val_score(
                        model, self.X_train_scaled, self.y_train, 
                        cv=cv, scoring=metric, n_jobs=-1
                    )
                    kernel_results[metric] = scores
                
                cv_results[kernel] = kernel_results
            
            # Display CV results
            st.subheader("Cross-Validation Results Summary")
            
            summary_data = []
            for kernel, kernel_results in cv_results.items():
                for metric, scores in kernel_results.items():
                    summary_data.append({
                        'Kernel': kernel.upper(),
                        'Metric': metric.upper(),
                        'Mean': f"{scores.mean():.4f}",
                        'Std': f"{scores.std():.4f}",
                        'Min': f"{scores.min():.4f}",
                        'Max': f"{scores.max():.4f}"
                    })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Visualize CV results
            st.subheader("Cross-Validation Scores Distribution")
            
            for metric in scoring_metrics:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                metric_data = []
                labels = []
                
                for kernel in cv_results.keys():
                    metric_data.append(cv_results[kernel][metric])
                    labels.append(kernel.upper())
                
                ax.boxplot(metric_data, labels=labels)
                ax.set_title(f'{metric.upper()} Scores Distribution')
                ax.set_ylabel('Score')
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)

def main():
    """Main Streamlit application."""
    
    # Title and description
    st.markdown('<h1 class="main-header">⚡ Support Vector Machine Classifier</h1>', unsafe_allow_html=True)
    st.markdown("**Interactive Machine Learning Application for Binary Classification with SVM**")
    
    # Initialize classifier
    classifier = StreamlitSVMClassifier()
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Control Panel")
    
    # Data loading options
    st.sidebar.markdown("### 📁 Data Loading")
    data_option = st.sidebar.radio(
        "Choose data source:",
        ["Upload CSV File", "Use Default Churn Dataset"],
        key="svm_data_option"
    )
    
    if data_option == "Upload CSV File":
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv", key="svm_upload")
        if uploaded_file is not None:
            if classifier.load_data(uploaded_file=uploaded_file):
                st.sidebar.success("✅ Data loaded successfully!")
    else:
        default_path = r"c:\Users\ITAPPS002\OneDrive\Documents\Data Set For Task\Churn Prdiction Data\churn-bigml-80.csv"
        if st.sidebar.button("Load Default Dataset", key="svm_load_default"):
            if classifier.load_data(file_path=default_path):
                st.sidebar.success("✅ Default dataset loaded!")
    
    # Main content
    if st.session_state.svm_data_loaded and classifier.df is not None:
        
        # Data overview
        classifier.display_data_overview()
        
        # Preprocessing
        st.markdown('<h2 class="sub-header">🔧 Data Preprocessing</h2>', unsafe_allow_html=True)
        if st.button("Preprocess Data", key="svm_preprocess"):
            classifier.preprocess_data()
        
        # Model training
        if (classifier.X_train_scaled is not None or 
            st.session_state.svm_X_train_scaled is not None):
            classifier.train_svm_models()
        
        # Model evaluation and analysis
        if st.session_state.svm_models_trained:
            
            # Evaluation tabs
            eval_tab1, eval_tab2, eval_tab3, eval_tab4 = st.tabs(["📊 Model Evaluation", "🎯 Decision Boundary", "🔄 Cross-Validation", "📈 Kernel Comparison"])
            
            with eval_tab1:
                classifier.evaluate_models()
            
            with eval_tab2:
                classifier.visualize_decision_boundary()
            
            with eval_tab3:
                classifier.cross_validation_analysis()
            
            with eval_tab4:
                st.markdown('<h2 class="sub-header">📈 Kernel Performance Comparison</h2>', unsafe_allow_html=True)
                
                if st.session_state.svm_models:
                    # Performance summary
                    st.subheader("Performance Summary")
                    
                    performance_data = []
                    for kernel, model_info in st.session_state.svm_models.items():
                        model = model_info['model']
                        
                        # Load test data from session state
                        X_test_scaled = st.session_state.svm_X_test_scaled
                        y_test = st.session_state.svm_y_test
                        
                        if X_test_scaled is not None and y_test is not None:
                            y_pred = model.predict(X_test_scaled)
                            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                            
                            performance_data.append({
                                'Kernel': kernel.upper(),
                                'Accuracy': accuracy_score(y_test, y_pred),
                                'Precision': precision_score(y_test, y_pred),
                                'Recall': recall_score(y_test, y_pred),
                                'F1-Score': f1_score(y_test, y_pred),
                                'AUC': roc_auc_score(y_test, y_pred_proba),
                                'Training Time (s)': model_info['training_time']
                            })
                    
                    if performance_data:
                        perf_df = pd.DataFrame(performance_data)
                        st.dataframe(perf_df.round(4), use_container_width=True)
                        
                        # Best performing model
                        best_f1_idx = perf_df['F1-Score'].idxmax()
                        best_model = perf_df.iloc[best_f1_idx]
                        
                        st.success(f"🏆 Best performing model: **{best_model['Kernel']} Kernel** (F1-Score: {best_model['F1-Score']:.4f})")
    
    else:
        st.info("👆 Please load a dataset to get started!")
        
        # Show sample data format
        st.markdown("### 📝 Expected Data Format")
        st.markdown("""
        Your CSV file should contain:
        - A target column named 'Churn' (True/False or 1/0) for binary classification
        - Feature columns with numerical or categorical data
        - No missing values in critical columns
        
        **SVM Benefits:**
        - Effective in high-dimensional spaces
        - Memory efficient (uses support vectors)
        - Versatile (different kernel functions)
        - Works well with clear margin separation
        """)

if __name__ == "__main__":
    main()
