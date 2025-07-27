"""
Visualization Utilities for Ship Steering Gear Fault Diagnosis
Comprehensive plotting functions for data analysis and results visualization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class DataVisualizer:
    """Comprehensive data visualization tools"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
    def plot_signal_samples(self, data: pd.DataFrame, signal_columns: List[str],
                           conditions: Optional[List[str]] = None, 
                           n_samples: int = 3, save_path: Optional[str] = None) -> None:
        """
        Plot sample signals for different conditions
        
        Args:
            data: DataFrame containing signal data
            signal_columns: List of signal column names
            conditions: List of conditions to plot
            n_samples: Number of samples per condition
            save_path: Path to save the plot
        """
        if conditions is None:
            conditions = data['condition'].unique()
        
        n_signals = len(signal_columns)
        n_conditions = len(conditions)
        
        fig, axes = plt.subplots(n_signals, n_conditions, 
                                figsize=(4*n_conditions, 3*n_signals))
        
        if n_signals == 1:
            axes = axes.reshape(1, -1)
        if n_conditions == 1:
            axes = axes.reshape(-1, 1)
        
        for i, signal_col in enumerate(signal_columns):
            for j, condition in enumerate(conditions):
                # Get samples for this condition
                condition_data = data[data['condition'] == condition]
                samples = condition_data.head(n_samples)
                
                ax = axes[i, j] if n_signals > 1 else axes[j]
                
                for idx, (_, sample) in enumerate(samples.iterrows()):
                    if signal_col in sample and isinstance(sample[signal_col], np.ndarray):
                        signal = sample[signal_col]
                        time = np.linspace(0, len(signal)/100, len(signal))  # Assuming 100 Hz
                        ax.plot(time, signal, alpha=0.7, label=f'Sample {idx+1}')
                
                ax.set_title(f'{signal_col} - {condition}')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.grid(True, alpha=0.3)
                if j == 0:  # Only show legend for first column
                    ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_distributions(self, df: pd.DataFrame, features: List[str],
                                  target_column: str = 'label', 
                                  save_path: Optional[str] = None) -> None:
        """
        Plot feature distributions by class
        
        Args:
            df: DataFrame with features
            features: List of feature names to plot
            target_column: Target column name
            save_path: Path to save the plot
        """
        n_features = len(features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, feature in enumerate(features):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Plot distribution for each class
            for label in sorted(df[target_column].unique()):
                data = df[df[target_column] == label][feature].dropna()
                ax.hist(data, alpha=0.6, label=f'Class {label}', bins=20)
            
            ax.set_title(f'Distribution of {feature}')
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_features, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, features: List[str],
                                figsize: Optional[Tuple[int, int]] = None,
                                save_path: Optional[str] = None) -> None:
        """
        Plot correlation heatmap of features
        
        Args:
            df: DataFrame with features
            features: List of feature names
            figsize: Figure size
            save_path: Path to save the plot
        """
        if figsize is None:
            figsize = (max(8, len(features)//2), max(6, len(features)//2))
        
        # Calculate correlation matrix
        corr_matrix = df[features].corr()
        
        plt.figure(figsize=figsize)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Plot heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, 
                   cbar_kws={"shrink": 0.8})
        
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, results_df: pd.DataFrame, 
                             metrics: List[str] = ['accuracy', 'f1_macro'],
                             save_path: Optional[str] = None) -> None:
        """
        Plot model comparison results
        
        Args:
            results_df: DataFrame with model comparison results
            metrics: List of metrics to plot
            save_path: Path to save the plot
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if metric in results_df.columns:
                bars = axes[i].bar(results_df['Model'], results_df[metric], 
                                  color=self.colors[:len(results_df)])
                axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
                axes[i].set_ylabel(metric.replace("_", " ").title())
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_optimization_convergence(self, history: List[Dict], 
                                    save_path: Optional[str] = None) -> None:
        """
        Plot optimization convergence history
        
        Args:
            history: List of optimization history dictionaries
            save_path: Path to save the plot
        """
        history_df = pd.DataFrame(history)
        
        plt.figure(figsize=self.figsize)
        plt.plot(history_df['iteration'], history_df['best_score'], 
                'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Iteration')
        plt.ylabel('Best Score')
        plt.title(f'{history_df["algorithm"].iloc[0]} Optimization Convergence')
        plt.grid(True, alpha=0.3)
        
        # Add final score annotation
        final_score = history_df['best_score'].iloc[-1]
        plt.annotate(f'Final Score: {final_score:.4f}', 
                    xy=(history_df['iteration'].iloc[-1], final_score),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 20,
                               save_path: Optional[str] = None) -> None:
        """
        Plot feature importance
        
        Args:
            importance_df: DataFrame with feature importance
            top_n: Number of top features to show
            save_path: Path to save the plot
        """
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, max(6, top_n//3)))
        bars = plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        
        # Color bars by importance
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_robustness_analysis(self, noise_results: Dict[float, Dict],
                                data_size_results: Dict[float, Dict],
                                save_path: Optional[str] = None) -> None:
        """
        Plot robustness analysis results
        
        Args:
            noise_results: Results from noise robustness evaluation
            data_size_results: Results from data size robustness evaluation
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Noise robustness plot
        snr_levels = list(noise_results.keys())
        f1_scores = [results['f1_macro'] for results in noise_results.values()]
        
        axes[0].plot(snr_levels, f1_scores, 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('SNR (dB)')
        axes[0].set_ylabel('F1-Score (Macro)')
        axes[0].set_title('Model Performance vs. Noise Level')
        axes[0].grid(True, alpha=0.3)
        
        # Add annotations
        for snr, f1 in zip(snr_levels, f1_scores):
            axes[0].annotate(f'{f1:.3f}', (snr, f1), 
                           textcoords="offset points", xytext=(0,10), ha='center')
        
        # Data size robustness plot
        data_ratios = list(data_size_results.keys())
        mean_f1_scores = [results['mean_f1'] for results in data_size_results.values()]
        std_f1_scores = [results['std_f1'] for results in data_size_results.values()]
        
        axes[1].errorbar(data_ratios, mean_f1_scores, yerr=std_f1_scores, 
                        fmt='ro-', linewidth=2, markersize=8, capsize=5)
        axes[1].set_xlabel('Training Data Ratio')
        axes[1].set_ylabel('F1-Score (Macro)')
        axes[1].set_title('Model Performance vs. Training Data Size')
        axes[1].grid(True, alpha=0.3)
        
        # Add annotations
        for ratio, f1, std in zip(data_ratios, mean_f1_scores, std_f1_scores):
            axes[1].annotate(f'{f1:.3f}±{std:.3f}', (ratio, f1), 
                           textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices_comparison(self, confusion_matrices: Dict[str, np.ndarray],
                                         class_names: List[str],
                                         save_path: Optional[str] = None) -> None:
        """
        Plot multiple confusion matrices for comparison
        
        Args:
            confusion_matrices: Dictionary of confusion matrices by model name
            class_names: List of class names
            save_path: Path to save the plot
        """
        n_models = len(confusion_matrices)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (model_name, cm) in enumerate(confusion_matrices.items()):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_title(f'{model_name}')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
        
        # Hide empty subplots
        for i in range(n_models, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def create_summary_report(results: Dict[str, Any], save_path: Optional[str] = None) -> str:
    """
    Create a comprehensive summary report
    
    Args:
        results: Dictionary containing all experimental results
        save_path: Path to save the report
        
    Returns:
        Summary report as string
    """
    report = []
    report.append("=" * 80)
    report.append("SHIP STEERING GEAR FAULT DIAGNOSIS - EXPERIMENTAL RESULTS SUMMARY")
    report.append("=" * 80)
    report.append("")
    
    # Dataset information
    if 'dataset_info' in results:
        info = results['dataset_info']
        report.append("DATASET INFORMATION:")
        report.append(f"  Total samples: {info.get('total_samples', 'N/A')}")
        report.append(f"  Number of features: {info.get('n_features', 'N/A')}")
        report.append(f"  Number of classes: {info.get('n_classes', 'N/A')}")
        report.append(f"  Class distribution: {info.get('class_distribution', 'N/A')}")
        report.append("")
    
    # Model performance
    if 'model_comparison' in results:
        report.append("MODEL PERFORMANCE COMPARISON:")
        comparison_df = results['model_comparison']
        report.append(comparison_df.to_string(index=False))
        report.append("")
    
    # Best model details
    if 'best_model' in results:
        best = results['best_model']
        report.append("BEST MODEL DETAILS:")
        report.append(f"  Model: {best.get('name', 'N/A')}")
        report.append(f"  Accuracy: {best.get('accuracy', 'N/A'):.4f}")
        report.append(f"  F1-Score (Macro): {best.get('f1_macro', 'N/A'):.4f}")
        if 'hyperparameters' in best:
            report.append("  Optimized Hyperparameters:")
            for param, value in best['hyperparameters'].items():
                report.append(f"    {param}: {value}")
        report.append("")
    
    # Robustness analysis
    if 'robustness' in results:
        robustness = results['robustness']
        report.append("ROBUSTNESS ANALYSIS:")
        
        if 'noise_robustness' in robustness:
            report.append("  Noise Robustness (F1-Score vs SNR):")
            for snr, metrics in robustness['noise_robustness'].items():
                report.append(f"    SNR {snr} dB: {metrics['f1_macro']:.4f}")
        
        if 'data_size_robustness' in robustness:
            report.append("  Data Size Robustness (F1-Score vs Training Size):")
            for ratio, metrics in robustness['data_size_robustness'].items():
                report.append(f"    {ratio*100:.0f}% data: {metrics['mean_f1']:.4f} ± {metrics['std_f1']:.4f}")
        report.append("")
    
    # Feature analysis
    if 'feature_analysis' in results:
        feature_analysis = results['feature_analysis']
        report.append("FEATURE ANALYSIS:")
        report.append(f"  Original features: {feature_analysis.get('original_features', 'N/A')}")
        report.append(f"  Selected features: {feature_analysis.get('selected_features', 'N/A')}")
        report.append(f"  Feature selection method: {feature_analysis.get('selection_method', 'N/A')}")
        if 'top_features' in feature_analysis:
            report.append("  Top 10 Important Features:")
            for i, feature in enumerate(feature_analysis['top_features'][:10], 1):
                report.append(f"    {i}. {feature}")
        report.append("")
    
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
    
    return report_text
