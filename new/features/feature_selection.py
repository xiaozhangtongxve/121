"""
Feature Selection and Analysis for Ship Steering Gear Fault Diagnosis
Implements correlation analysis, dimensionality reduction, and feature visualization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class FeatureSelector:
    """Feature selection and analysis tools"""
    
    def __init__(self, correlation_threshold: float = 0.95):
        """
        Initialize feature selector
        
        Args:
            correlation_threshold: Threshold for removing highly correlated features
        """
        self.correlation_threshold = correlation_threshold
        self.selected_features = None
        self.correlation_matrix = None
        self.feature_importance = None
        
    def remove_correlated_features(self, df: pd.DataFrame, 
                                 target_column: str = 'label') -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove highly correlated features using Pearson correlation
        
        Args:
            df: DataFrame with features
            target_column: Name of target column
            
        Returns:
            Tuple of (filtered DataFrame, list of removed features)
        """
        # Separate features from target
        feature_columns = [col for col in df.columns if col not in 
                          [target_column, 'sample_id', 'condition', 'noise_level']]
        
        # Calculate correlation matrix
        feature_df = df[feature_columns]
        self.correlation_matrix = feature_df.corr().abs()
        
        # Find highly correlated feature pairs
        upper_triangle = self.correlation_matrix.where(
            np.triu(np.ones(self.correlation_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to remove
        to_remove = [column for column in upper_triangle.columns 
                    if any(upper_triangle[column] > self.correlation_threshold)]
        
        # Keep features not in removal list
        features_to_keep = [col for col in feature_columns if col not in to_remove]
        
        # Create filtered DataFrame
        filtered_df = df[['sample_id', 'condition', 'noise_level', target_column] + features_to_keep].copy()
        
        self.selected_features = features_to_keep
        
        print(f"Removed {len(to_remove)} highly correlated features")
        print(f"Kept {len(features_to_keep)} features")
        
        return filtered_df, to_remove
    
    def select_k_best_features(self, df: pd.DataFrame, k: int = 50, 
                             target_column: str = 'label', 
                             method: str = 'f_classif') -> Tuple[pd.DataFrame, List[str]]:
        """
        Select k best features using statistical tests
        
        Args:
            df: DataFrame with features
            k: Number of features to select
            target_column: Name of target column
            method: Selection method ('f_classif' or 'mutual_info')
            
        Returns:
            Tuple of (filtered DataFrame, list of selected features)
        """
        # Separate features from target
        feature_columns = [col for col in df.columns if col not in 
                          [target_column, 'sample_id', 'condition', 'noise_level']]
        
        X = df[feature_columns]
        y = df[target_column]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Select scoring function
        if method == 'f_classif':
            score_func = f_classif
        elif method == 'mutual_info':
            score_func = mutual_info_classif
        else:
            raise ValueError("Method must be 'f_classif' or 'mutual_info'")
        
        # Select k best features
        selector = SelectKBest(score_func=score_func, k=min(k, len(feature_columns)))
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = [feature_columns[i] for i, selected in enumerate(selected_mask) if selected]
        
        # Get feature scores
        scores = selector.scores_
        self.feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'score': scores,
            'selected': selected_mask
        }).sort_values('score', ascending=False)
        
        # Create filtered DataFrame
        filtered_df = df[['sample_id', 'condition', 'noise_level', target_column] + selected_features].copy()
        
        print(f"Selected {len(selected_features)} best features using {method}")
        
        return filtered_df, selected_features
    
    def plot_correlation_matrix(self, df: pd.DataFrame, figsize: Tuple[int, int] = (12, 10),
                               save_path: Optional[str] = None) -> None:
        """Plot correlation matrix heatmap"""
        feature_columns = [col for col in df.columns if col not in 
                          ['label', 'sample_id', 'condition', 'noise_level']]
        
        # Calculate correlation matrix
        corr_matrix = df[feature_columns].corr()
        
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (10, 8),
                               save_path: Optional[str] = None) -> None:
        """Plot feature importance scores"""
        if self.feature_importance is None:
            print("No feature importance data available. Run select_k_best_features first.")
            return
        
        top_features = self.feature_importance.head(top_n)
        
        plt.figure(figsize=figsize)
        sns.barplot(data=top_features, x='score', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance Scores')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class DimensionalityReducer:
    """Dimensionality reduction and visualization tools"""
    
    def __init__(self):
        self.pca_model = None
        self.tsne_model = None
        self.scaler = StandardScaler()
    
    def apply_pca(self, df: pd.DataFrame, n_components: int = 2, 
                  target_column: str = 'label') -> Tuple[np.ndarray, PCA]:
        """
        Apply PCA for dimensionality reduction
        
        Args:
            df: DataFrame with features
            n_components: Number of principal components
            target_column: Name of target column
            
        Returns:
            Tuple of (transformed data, PCA model)
        """
        # Separate features from target
        feature_columns = [col for col in df.columns if col not in 
                          [target_column, 'sample_id', 'condition', 'noise_level']]
        
        X = df[feature_columns]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA
        self.pca_model = PCA(n_components=n_components)
        X_pca = self.pca_model.fit_transform(X_scaled)
        
        return X_pca, self.pca_model
    
    def apply_tsne(self, df: pd.DataFrame, n_components: int = 2, 
                   target_column: str = 'label', perplexity: int = 30,
                   random_state: int = 42) -> np.ndarray:
        """
        Apply t-SNE for dimensionality reduction
        
        Args:
            df: DataFrame with features
            n_components: Number of t-SNE components
            target_column: Name of target column
            perplexity: t-SNE perplexity parameter
            random_state: Random state for reproducibility
            
        Returns:
            Transformed data
        """
        # Separate features from target
        feature_columns = [col for col in df.columns if col not in 
                          [target_column, 'sample_id', 'condition', 'noise_level']]
        
        X = df[feature_columns]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply t-SNE
        self.tsne_model = TSNE(n_components=n_components, perplexity=perplexity, 
                              random_state=random_state, n_iter=1000)
        X_tsne = self.tsne_model.fit_transform(X_scaled)
        
        return X_tsne
    
    def plot_pca_analysis(self, df: pd.DataFrame, target_column: str = 'label',
                         figsize: Tuple[int, int] = (15, 5), 
                         save_path: Optional[str] = None) -> None:
        """Plot PCA analysis including explained variance and 2D visualization"""
        
        # Apply PCA with more components for variance analysis
        X_pca_full, pca_full = self.apply_pca(df, n_components=min(10, len(df.columns)-4), 
                                             target_column=target_column)
        
        # Apply PCA with 2 components for visualization
        X_pca_2d, _ = self.apply_pca(df, n_components=2, target_column=target_column)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot explained variance ratio
        axes[0].bar(range(1, len(pca_full.explained_variance_ratio_) + 1), 
                   pca_full.explained_variance_ratio_)
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('PCA Explained Variance')
        
        # Plot cumulative explained variance
        cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
        axes[1].plot(range(1, len(cumsum_var) + 1), cumsum_var, 'bo-')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Explained Variance')
        axes[1].set_title('Cumulative Explained Variance')
        axes[1].grid(True)
        
        # Plot 2D PCA visualization
        labels = df[target_column]
        unique_labels = sorted(labels.unique())
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            axes[2].scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], 
                           c=[colors[i]], label=f'Class {label}', alpha=0.7)
        
        axes[2].set_xlabel(f'PC1 ({pca_full.explained_variance_ratio_[0]:.2%} variance)')
        axes[2].set_ylabel(f'PC2 ({pca_full.explained_variance_ratio_[1]:.2%} variance)')
        axes[2].set_title('PCA 2D Visualization')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_tsne_visualization(self, df: pd.DataFrame, target_column: str = 'label',
                               figsize: Tuple[int, int] = (10, 8),
                               save_path: Optional[str] = None) -> None:
        """Plot t-SNE 2D visualization"""
        
        X_tsne = self.apply_tsne(df, target_column=target_column)
        
        plt.figure(figsize=figsize)
        
        labels = df[target_column]
        unique_labels = sorted(labels.unique())
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       c=[colors[i]], label=f'Class {label}', alpha=0.7, s=50)
        
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('t-SNE 2D Visualization of Feature Space')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_separability(self, df: pd.DataFrame, target_column: str = 'label',
                                 figsize: Tuple[int, int] = (15, 5),
                                 save_path: Optional[str] = None) -> None:
        """Plot both PCA and t-SNE visualizations for feature separability analysis"""
        
        X_pca = self.apply_pca(df, n_components=2, target_column=target_column)[0]
        X_tsne = self.apply_tsne(df, target_column=target_column)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        labels = df[target_column]
        unique_labels = sorted(labels.unique())
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        # PCA plot
        for i, label in enumerate(unique_labels):
            mask = labels == label
            axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                           c=[colors[i]], label=f'Class {label}', alpha=0.7)
        
        axes[0].set_xlabel('Principal Component 1')
        axes[0].set_ylabel('Principal Component 2')
        axes[0].set_title('PCA Feature Separability')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # t-SNE plot
        for i, label in enumerate(unique_labels):
            mask = labels == label
            axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                           c=[colors[i]], label=f'Class {label}', alpha=0.7)
        
        axes[1].set_xlabel('t-SNE Component 1')
        axes[1].set_ylabel('t-SNE Component 2')
        axes[1].set_title('t-SNE Feature Separability')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
