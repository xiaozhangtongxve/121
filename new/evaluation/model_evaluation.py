"""
Model Evaluation and Comparison for Ship Steering Gear Fault Diagnosis
Implements comprehensive evaluation metrics, confusion matrices, and comparative analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report, roc_curve, auc,
                           precision_recall_curve)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Comprehensive model evaluation and comparison tools"""
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize model evaluator
        
        Args:
            class_names: List of class names for visualization
        """
        self.class_names = class_names or ['Normal', 'Air Contamination', 'Internal Leakage', 
                                          'Valve Sticking', 'Filter Blockage']
        self.evaluation_results = {}
        
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                      model_name: str = 'Model') -> Dict[str, float]:
        """
        Evaluate a single model and return comprehensive metrics
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model for identification
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # Per-class metrics
        precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            if i < len(precision_per_class):
                metrics[f'precision_{class_name}'] = precision_per_class[i]
                metrics[f'recall_{class_name}'] = recall_per_class[i]
                metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        # Store results
        self.evaluation_results[model_name] = {
            'metrics': metrics,
            'y_true': y_test,
            'y_pred': y_pred,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return metrics
    
    def plot_confusion_matrix(self, model_name: str, figsize: Tuple[int, int] = (8, 6),
                             save_path: Optional[str] = None) -> None:
        """Plot confusion matrix for a specific model"""
        
        if model_name not in self.evaluation_results:
            print(f"No evaluation results found for {model_name}")
            return
        
        cm = self.evaluation_results[model_name]['confusion_matrix']
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_classification_report(self, model_name: str, figsize: Tuple[int, int] = (10, 6),
                                  save_path: Optional[str] = None) -> None:
        """Plot classification report as heatmap"""
        
        if model_name not in self.evaluation_results:
            print(f"No evaluation results found for {model_name}")
            return
        
        y_true = self.evaluation_results[model_name]['y_true']
        y_pred = self.evaluation_results[model_name]['y_pred']
        
        # Get classification report as dictionary
        report = classification_report(y_true, y_pred, target_names=self.class_names, 
                                     output_dict=True, zero_division=0)
        
        # Convert to DataFrame for visualization
        report_df = pd.DataFrame(report).iloc[:-1, :].T  # Exclude 'accuracy' row
        
        plt.figure(figsize=figsize)
        sns.heatmap(report_df.iloc[:-3, :3], annot=True, cmap='Blues', fmt='.3f')
        plt.title(f'Classification Report - {model_name}')
        plt.xlabel('Metrics')
        plt.ylabel('Classes')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_models(self, figsize: Tuple[int, int] = (12, 8),
                      save_path: Optional[str] = None) -> pd.DataFrame:
        """Compare multiple models and create visualization"""
        
        if len(self.evaluation_results) < 2:
            print("Need at least 2 models for comparison")
            return pd.DataFrame()
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in self.evaluation_results.items():
            metrics = results['metrics']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision (Macro)': metrics['precision_macro'],
                'Recall (Macro)': metrics['recall_macro'],
                'F1-Score (Macro)': metrics['f1_macro'],
                'Precision (Weighted)': metrics['precision_weighted'],
                'Recall (Weighted)': metrics['recall_weighted'],
                'F1-Score (Weighted)': metrics['f1_weighted']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Accuracy comparison
        axes[0, 0].bar(comparison_df['Model'], comparison_df['Accuracy'])
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Precision comparison
        axes[0, 1].bar(comparison_df['Model'], comparison_df['Precision (Macro)'])
        axes[0, 1].set_title('Precision Comparison (Macro)')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Recall comparison
        axes[1, 0].bar(comparison_df['Model'], comparison_df['Recall (Macro)'])
        axes[1, 0].set_title('Recall Comparison (Macro)')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # F1-Score comparison
        axes[1, 1].bar(comparison_df['Model'], comparison_df['F1-Score (Macro)'])
        axes[1, 1].set_title('F1-Score Comparison (Macro)')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return comparison_df
    
    def plot_per_class_performance(self, figsize: Tuple[int, int] = (15, 5),
                                  save_path: Optional[str] = None) -> None:
        """Plot per-class performance comparison across models"""
        
        if len(self.evaluation_results) == 0:
            print("No evaluation results available")
            return
        
        # Prepare data for per-class comparison
        metrics_data = {'Precision': {}, 'Recall': {}, 'F1-Score': {}}
        
        for model_name, results in self.evaluation_results.items():
            metrics = results['metrics']
            for class_name in self.class_names:
                if f'precision_{class_name}' in metrics:
                    if class_name not in metrics_data['Precision']:
                        metrics_data['Precision'][class_name] = {}
                        metrics_data['Recall'][class_name] = {}
                        metrics_data['F1-Score'][class_name] = {}
                    
                    metrics_data['Precision'][class_name][model_name] = metrics[f'precision_{class_name}']
                    metrics_data['Recall'][class_name][model_name] = metrics[f'recall_{class_name}']
                    metrics_data['F1-Score'][class_name][model_name] = metrics[f'f1_{class_name}']
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        for i, (metric_name, metric_data) in enumerate(metrics_data.items()):
            # Convert to DataFrame for easier plotting
            metric_df = pd.DataFrame(metric_data)
            
            # Plot grouped bar chart
            metric_df.plot(kind='bar', ax=axes[i], width=0.8)
            axes[i].set_title(f'{metric_name} by Class')
            axes[i].set_ylabel(metric_name)
            axes[i].set_xlabel('Model')
            axes[i].legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class BaselineComparator:
    """Compare optimized LightGBM with baseline models"""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize baseline comparator
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def prepare_baseline_models(self) -> Dict[str, Any]:
        """Prepare baseline models for comparison"""
        
        self.models = {
            'SVM': SVC(random_state=self.random_state, probability=True),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'XGBoost': xgb.XGBClassifier(random_state=self.random_state, eval_metric='mlogloss'),
            'LightGBM (Default)': lgb.LGBMClassifier(random_state=self.random_state, verbose=-1)
        }
        
        return self.models
    
    def train_and_evaluate_baselines(self, X_train: np.ndarray, X_test: np.ndarray,
                                   y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """
        Train and evaluate all baseline models
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation results for each model
        """
        # Prepare data
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        # Prepare baseline models
        self.prepare_baseline_models()
        
        # Train and evaluate each model
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            
            try:
                # Train model
                model.fit(X_train_scaled, y_train_encoded)
                
                # Make predictions
                y_pred_encoded = model.predict(X_test_scaled)
                y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
                    'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
                    'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
                    'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                }
                
                results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'y_pred': y_pred,
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }
                
                print(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1_macro']:.4f}")
                
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                results[model_name] = None
        
        return results


class RobustnessEvaluator:
    """Evaluate model robustness under different conditions"""
    
    def __init__(self):
        pass
    
    def add_noise_to_data(self, X: np.ndarray, snr_db: float) -> np.ndarray:
        """
        Add noise to data based on Signal-to-Noise Ratio
        
        Args:
            X: Input data
            snr_db: Signal-to-Noise Ratio in dB
            
        Returns:
            Noisy data
        """
        # Calculate signal power
        signal_power = np.mean(X ** 2, axis=0)
        
        # Calculate noise power based on SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Generate noise
        noise = np.random.normal(0, np.sqrt(noise_power), X.shape)
        
        return X + noise
    
    def evaluate_noise_robustness(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                                snr_levels: List[float] = [30, 20, 10]) -> Dict[float, Dict]:
        """
        Evaluate model performance under different noise levels
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            snr_levels: List of SNR levels in dB
            
        Returns:
            Dictionary of results for each SNR level
        """
        results = {}
        
        for snr in snr_levels:
            print(f"Evaluating at SNR = {snr} dB...")
            
            # Add noise to test data
            X_test_noisy = self.add_noise_to_data(X_test, snr)
            
            # Make predictions
            y_pred = model.predict(X_test_noisy)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
                'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
                'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0)
            }
            
            results[snr] = metrics
        
        return results
    
    def evaluate_data_size_robustness(self, model_class: Any, X: np.ndarray, y: np.ndarray,
                                    data_ratios: List[float] = [1.0, 0.8, 0.5, 0.3],
                                    cv_folds: int = 5) -> Dict[float, Dict]:
        """
        Evaluate model performance with different training data sizes
        
        Args:
            model_class: Model class to instantiate
            X: Full feature matrix
            y: Full target vector
            data_ratios: List of data size ratios
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary of results for each data ratio
        """
        results = {}
        
        for ratio in data_ratios:
            print(f"Evaluating with {ratio*100:.0f}% of training data...")
            
            # Sample data
            if ratio < 1.0:
                X_sample, _, y_sample, _ = train_test_split(X, y, train_size=ratio, 
                                                          stratify=y, random_state=42)
            else:
                X_sample, y_sample = X, y
            
            # Perform cross-validation
            model = model_class()
            cv_scores = cross_val_score(model, X_sample, y_sample, 
                                      cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                                      scoring='f1_macro')
            
            results[ratio] = {
                'mean_f1': np.mean(cv_scores),
                'std_f1': np.std(cv_scores),
                'cv_scores': cv_scores
            }
        
        return results
