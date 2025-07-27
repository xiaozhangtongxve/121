"""
Main Experimental Script for Ship Steering Gear Fault Diagnosis
Comprehensive implementation of the research methodology
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data.data_simulator import SteeringGearSimulator
from features.time_domain import TimeDomainFeatures
from features.frequency_domain import FrequencyDomainFeatures
from features.time_frequency_domain import WaveletPacketFeatures
from features.feature_selection import FeatureSelector, DimensionalityReducer
from models.optimized_lightgbm import OptimizedLightGBM
from evaluation.model_evaluation import ModelEvaluator, BaselineComparator, RobustnessEvaluator
from utils.visualization import DataVisualizer, create_summary_report

from sklearn.model_selection import train_test_split
import joblib


class SteeringGearExperiment:
    """Main experimental class for comprehensive fault diagnosis research"""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize experiment
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.results = {}
        
        # Initialize components
        self.simulator = SteeringGearSimulator(sampling_rate=100.0, duration=10.0)
        self.time_features = TimeDomainFeatures()
        self.freq_features = FrequencyDomainFeatures(sampling_rate=100.0)
        self.wpt_features = WaveletPacketFeatures(wavelet='db4', levels=4)
        self.feature_selector = FeatureSelector(correlation_threshold=0.95)
        self.dim_reducer = DimensionalityReducer()
        self.visualizer = DataVisualizer()
        
        # Signal columns
        self.signal_columns = [
            'pump_outlet_pressure', 'cylinder_a_pressure', 'cylinder_b_pressure',
            'piston_displacement', 'servo_valve_current'
        ]
        
        # Class names
        self.class_names = ['Normal', 'Air Contamination', 'Internal Leakage', 
                           'Valve Sticking', 'Filter Blockage']
    
    def step1_data_generation(self, samples_per_condition: int = 100) -> pd.DataFrame:
        """
        Step 1: Generate simulation data for all fault conditions
        
        Args:
            samples_per_condition: Number of samples per condition
            
        Returns:
            Generated dataset
        """
        print("=" * 60)
        print("STEP 1: DATA GENERATION")
        print("=" * 60)
        
        print(f"Generating {samples_per_condition} samples per condition...")
        
        # Generate dataset with different noise levels
        dataset = self.simulator.generate_dataset(
            samples_per_condition=samples_per_condition,
            noise_levels=[0.02, 0.03, 0.04]  # Different noise levels for robustness
        )
        
        print(f"Generated dataset with {len(dataset)} total samples")
        print(f"Class distribution:")
        print(dataset['condition'].value_counts())
        
        # Store dataset info
        self.results['dataset_info'] = {
            'total_samples': len(dataset),
            'samples_per_condition': samples_per_condition,
            'n_classes': len(self.class_names),
            'class_distribution': dataset['condition'].value_counts().to_dict()
        }
        
        # Visualize sample signals
        print("\nVisualizing sample signals...")
        self.visualizer.plot_signal_samples(
            dataset, self.signal_columns, n_samples=2,
            save_path='results/sample_signals.png'
        )
        
        return dataset
    
    def step2_feature_extraction(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Extract multi-domain features
        
        Args:
            dataset: Raw dataset with signals
            
        Returns:
            Dataset with extracted features
        """
        print("\n" + "=" * 60)
        print("STEP 2: MULTI-DOMAIN FEATURE EXTRACTION")
        print("=" * 60)
        
        # Extract time domain features
        print("Extracting time domain features...")
        time_features_df = self.time_features.extract_features_from_dataframe(
            dataset, self.signal_columns
        )
        
        # Extract frequency domain features
        print("Extracting frequency domain features...")
        freq_features_df = self.freq_features.extract_features_from_dataframe(
            dataset, self.signal_columns
        )
        
        # Extract time-frequency domain features (WPT)
        print("Extracting time-frequency domain features...")
        wpt_features_df = self.wpt_features.extract_features_from_dataframe(
            dataset, self.signal_columns
        )
        
        # Combine all features
        print("Combining all features...")
        combined_features = time_features_df.copy()
        
        # Merge frequency features
        freq_feature_cols = [col for col in freq_features_df.columns 
                           if col not in ['sample_id', 'condition', 'label', 'noise_level']]
        for col in freq_feature_cols:
            combined_features[col] = freq_features_df[col]
        
        # Merge WPT features
        wpt_feature_cols = [col for col in wpt_features_df.columns 
                          if col not in ['sample_id', 'condition', 'label', 'noise_level']]
        for col in wpt_feature_cols:
            combined_features[col] = wpt_features_df[col]
        
        # Handle missing values
        feature_cols = [col for col in combined_features.columns 
                       if col not in ['sample_id', 'condition', 'label', 'noise_level']]
        combined_features[feature_cols] = combined_features[feature_cols].fillna(0)
        
        print(f"Extracted {len(feature_cols)} total features")
        print(f"Feature breakdown:")
        print(f"  Time domain: {len(self.time_features.get_feature_names(self.signal_columns))}")
        print(f"  Frequency domain: {len(self.freq_features.get_feature_names(self.signal_columns))}")
        print(f"  Time-frequency domain: {len(wpt_feature_cols)}")
        
        # Store feature info
        self.results['feature_analysis'] = {
            'original_features': len(feature_cols),
            'time_domain_features': len(self.time_features.get_feature_names(self.signal_columns)),
            'frequency_domain_features': len(self.freq_features.get_feature_names(self.signal_columns)),
            'time_frequency_features': len(wpt_feature_cols)
        }
        
        return combined_features
    
    def step3_feature_selection(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 3: Feature selection and analysis
        
        Args:
            features_df: DataFrame with extracted features
            
        Returns:
            DataFrame with selected features
        """
        print("\n" + "=" * 60)
        print("STEP 3: FEATURE SELECTION AND ANALYSIS")
        print("=" * 60)
        
        # Remove highly correlated features
        print("Removing highly correlated features...")
        filtered_df, removed_features = self.feature_selector.remove_correlated_features(
            features_df, target_column='label'
        )
        
        # Select k best features
        print("Selecting best features using statistical tests...")
        selected_df, selected_features = self.feature_selector.select_k_best_features(
            filtered_df, k=50, target_column='label', method='f_classif'
        )
        
        # Visualize feature analysis
        print("Creating feature analysis visualizations...")
        
        # Plot feature importance
        self.feature_selector.plot_feature_importance(
            top_n=20, save_path='results/feature_importance.png'
        )
        
        # Plot correlation matrix for selected features
        feature_cols = [col for col in selected_df.columns 
                       if col not in ['sample_id', 'condition', 'label', 'noise_level']]
        if len(feature_cols) <= 50:  # Only plot if manageable number
            self.feature_selector.plot_correlation_matrix(
                selected_df[feature_cols + ['label']], 
                save_path='results/correlation_matrix.png'
            )
        
        # Dimensionality reduction visualization
        print("Creating dimensionality reduction visualizations...")
        self.dim_reducer.plot_feature_separability(
            selected_df, target_column='label',
            save_path='results/feature_separability.png'
        )
        
        # Update feature analysis results
        self.results['feature_analysis'].update({
            'selected_features': len(selected_features),
            'removed_correlated': len(removed_features),
            'selection_method': 'f_classif',
            'top_features': selected_features[:20]  # Top 20 features
        })
        
        return selected_df
    
    def step4_model_training(self, features_df: pd.DataFrame) -> OptimizedLightGBM:
        """
        Step 4: Train optimized LightGBM model
        
        Args:
            features_df: DataFrame with selected features
            
        Returns:
            Trained optimized LightGBM model
        """
        print("\n" + "=" * 60)
        print("STEP 4: OPTIMIZED LIGHTGBM MODEL TRAINING")
        print("=" * 60)
        
        # Prepare data
        feature_cols = [col for col in features_df.columns 
                       if col not in ['sample_id', 'condition', 'label', 'noise_level']]
        
        X = features_df[feature_cols].values
        y = features_df['label'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Initialize and train optimized LightGBM
        print("Training optimized LightGBM with PSO optimization...")
        optimized_lgb = OptimizedLightGBM(random_state=self.random_state)
        
        # Train with optimization
        optimized_lgb.fit(X_train, y_train, optimize=True, optimization_method='PSO')
        
        # Plot optimization convergence
        optimized_lgb.plot_optimization_convergence(
            save_path='results/optimization_convergence.png'
        )
        
        # Evaluate on test set
        y_pred = optimized_lgb.predict(X_test)
        
        # Store model results
        self.results['best_model'] = {
            'name': 'Optimized LightGBM (PSO)',
            'hyperparameters': optimized_lgb.best_params,
            'model': optimized_lgb
        }
        
        # Store test data for evaluation
        self.results['test_data'] = {
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': feature_cols
        }
        
        return optimized_lgb
    
    def step5_model_evaluation(self, optimized_model: OptimizedLightGBM) -> None:
        """
        Step 5: Comprehensive model evaluation and comparison
        
        Args:
            optimized_model: Trained optimized LightGBM model
        """
        print("\n" + "=" * 60)
        print("STEP 5: MODEL EVALUATION AND COMPARISON")
        print("=" * 60)
        
        # Get test data
        X_test = self.results['test_data']['X_test']
        y_test = self.results['test_data']['y_test']
        
        # Initialize evaluator
        evaluator = ModelEvaluator(class_names=self.class_names)
        
        # Evaluate optimized LightGBM
        print("Evaluating optimized LightGBM...")
        lgb_metrics = evaluator.evaluate_model(
            optimized_model, X_test, y_test, 'Optimized LightGBM (PSO)'
        )
        
        # Compare with baseline models
        print("Training and evaluating baseline models...")
        comparator = BaselineComparator(random_state=self.random_state)
        
        # Get training data for baseline comparison
        feature_cols = self.results['test_data']['feature_names']
        full_df = self.results.get('selected_features_df')
        if full_df is not None:
            X_full = full_df[feature_cols].values
            y_full = full_df['label'].values
            X_train_full, _, y_train_full, _ = train_test_split(
                X_full, y_full, test_size=0.2, stratify=y_full, random_state=self.random_state
            )
        else:
            # Use available test data for demonstration
            X_train_full, y_train_full = X_test, y_test
        
        baseline_results = comparator.train_and_evaluate_baselines(
            X_train_full, X_test, y_train_full, y_test
        )
        
        # Add baseline results to evaluator
        for model_name, result in baseline_results.items():
            if result is not None:
                evaluator.evaluation_results[model_name] = {
                    'metrics': result['metrics'],
                    'y_true': y_test,
                    'y_pred': result['y_pred'],
                    'confusion_matrix': result['confusion_matrix']
                }
        
        # Create comparison visualizations
        print("Creating evaluation visualizations...")
        
        # Model comparison
        comparison_df = evaluator.compare_models(save_path='results/model_comparison.png')
        
        # Confusion matrices
        evaluator.plot_confusion_matrix(
            'Optimized LightGBM (PSO)', save_path='results/confusion_matrix_optimized.png'
        )
        
        # Per-class performance
        evaluator.plot_per_class_performance(save_path='results/per_class_performance.png')
        
        # Store comparison results
        self.results['model_comparison'] = comparison_df
        self.results['best_model'].update(lgb_metrics)
        
    def step6_robustness_testing(self, optimized_model: OptimizedLightGBM) -> None:
        """
        Step 6: Robustness testing under different conditions
        
        Args:
            optimized_model: Trained optimized LightGBM model
        """
        print("\n" + "=" * 60)
        print("STEP 6: ROBUSTNESS TESTING")
        print("=" * 60)
        
        # Get test data
        X_test = self.results['test_data']['X_test']
        y_test = self.results['test_data']['y_test']
        
        # Initialize robustness evaluator
        robustness_eval = RobustnessEvaluator()
        
        # Test noise robustness
        print("Testing noise robustness...")
        noise_results = robustness_eval.evaluate_noise_robustness(
            optimized_model, X_test, y_test, snr_levels=[30, 20, 10]
        )
        
        # Test data size robustness
        print("Testing data size robustness...")
        feature_cols = self.results['test_data']['feature_names']
        full_df = self.results.get('selected_features_df')
        if full_df is not None:
            X_full = full_df[feature_cols].values
            y_full = full_df['label'].values
            
            data_size_results = robustness_eval.evaluate_data_size_robustness(
                lambda: OptimizedLightGBM(random_state=self.random_state),
                X_full, y_full, data_ratios=[1.0, 0.8, 0.5, 0.3]
            )
        else:
            data_size_results = {}
        
        # Visualize robustness results
        if noise_results and data_size_results:
            self.visualizer.plot_robustness_analysis(
                noise_results, data_size_results,
                save_path='results/robustness_analysis.png'
            )
        
        # Store robustness results
        self.results['robustness'] = {
            'noise_robustness': noise_results,
            'data_size_robustness': data_size_results
        }
    
    def run_complete_experiment(self, samples_per_condition: int = 100) -> Dict[str, Any]:
        """
        Run the complete experimental pipeline
        
        Args:
            samples_per_condition: Number of samples per condition
            
        Returns:
            Complete experimental results
        """
        print("STARTING COMPLETE SHIP STEERING GEAR FAULT DIAGNOSIS EXPERIMENT")
        print("=" * 80)
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # Step 1: Data Generation
        dataset = self.step1_data_generation(samples_per_condition)
        
        # Step 2: Feature Extraction
        features_df = self.step2_feature_extraction(dataset)
        
        # Step 3: Feature Selection
        selected_features_df = self.step3_feature_selection(features_df)
        self.results['selected_features_df'] = selected_features_df
        
        # Step 4: Model Training
        optimized_model = self.step4_model_training(selected_features_df)
        
        # Step 5: Model Evaluation
        self.step5_model_evaluation(optimized_model)
        
        # Step 6: Robustness Testing
        self.step6_robustness_testing(optimized_model)
        
        # Generate summary report
        print("\n" + "=" * 60)
        print("GENERATING SUMMARY REPORT")
        print("=" * 60)
        
        summary_report = create_summary_report(self.results, 'results/experiment_summary.txt')
        print(summary_report)
        
        # Save complete results
        joblib.dump(self.results, 'results/complete_results.pkl')
        
        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("Results saved in 'results/' directory")
        print("=" * 80)
        
        return self.results


if __name__ == "__main__":
    # Run the complete experiment
    experiment = SteeringGearExperiment(random_state=42)
    results = experiment.run_complete_experiment(samples_per_condition=50)  # Reduced for demo
