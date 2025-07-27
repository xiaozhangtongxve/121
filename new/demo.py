"""
Quick Demo Script for Ship Steering Gear Fault Diagnosis
Demonstrates key functionality with a smaller dataset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data.data_simulator import SteeringGearSimulator
from features.time_domain import TimeDomainFeatures
from features.frequency_domain import FrequencyDomainFeatures
from models.optimized_lightgbm import OptimizedLightGBM
from evaluation.model_evaluation import ModelEvaluator
from utils.visualization import DataVisualizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report


def demo_data_simulation():
    """Demonstrate data simulation capabilities"""
    print("=" * 50)
    print("DEMO: Data Simulation")
    print("=" * 50)
    
    # Initialize simulator
    simulator = SteeringGearSimulator(sampling_rate=100.0, duration=5.0)  # Shorter duration for demo
    
    # Generate sample data for each condition
    print("Generating sample data for each fault condition...")
    
    conditions = {
        'Normal': simulator.generate_normal_condition(),
        'Air Contamination': simulator.generate_air_contamination_fault(),
        'Internal Leakage': simulator.generate_internal_leakage_fault(),
        'Valve Sticking': simulator.generate_valve_sticking_fault(),
        'Filter Blockage': simulator.generate_filter_blockage_fault()
    }
    
    # Plot sample signals
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    signal_names = ['pump_outlet_pressure', 'cylinder_a_pressure', 'cylinder_b_pressure',
                   'piston_displacement', 'servo_valve_current']
    
    for i, (condition_name, data) in enumerate(conditions.items()):
        if i < len(axes):
            # Plot pump outlet pressure as example
            time = np.linspace(0, 5, len(data['pump_outlet_pressure']))
            axes[i].plot(time, data['pump_outlet_pressure'])
            axes[i].set_title(f'{condition_name}')
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Pressure (bar)')
            axes[i].grid(True, alpha=0.3)
    
    # Hide the last subplot
    axes[-1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('demo_signals.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Sample signals plotted and saved as 'demo_signals.png'")
    return simulator


def demo_feature_extraction():
    """Demonstrate feature extraction"""
    print("\n" + "=" * 50)
    print("DEMO: Feature Extraction")
    print("=" * 50)
    
    # Generate small dataset
    simulator = SteeringGearSimulator(sampling_rate=100.0, duration=3.0)
    dataset = simulator.generate_dataset(samples_per_condition=10)  # Small dataset for demo
    
    signal_columns = ['pump_outlet_pressure', 'cylinder_a_pressure', 'cylinder_b_pressure',
                     'piston_displacement', 'servo_valve_current']
    
    # Extract time domain features
    print("Extracting time domain features...")
    time_extractor = TimeDomainFeatures()
    time_features = time_extractor.extract_features_from_dataframe(dataset, signal_columns)
    
    # Extract frequency domain features
    print("Extracting frequency domain features...")
    freq_extractor = FrequencyDomainFeatures(sampling_rate=100.0)
    freq_features = freq_extractor.extract_features_from_dataframe(dataset, signal_columns)
    
    # Combine features
    combined_features = time_features.copy()
    freq_feature_cols = [col for col in freq_features.columns 
                        if col not in ['sample_id', 'condition', 'label', 'noise_level']]
    for col in freq_feature_cols:
        combined_features[col] = freq_features[col]
    
    # Handle missing values
    feature_cols = [col for col in combined_features.columns 
                   if col not in ['sample_id', 'condition', 'label', 'noise_level']]
    combined_features[feature_cols] = combined_features[feature_cols].fillna(0)
    
    print(f"Extracted {len(feature_cols)} total features")
    print(f"Dataset shape: {combined_features.shape}")
    print(f"Class distribution:")
    print(combined_features['condition'].value_counts())
    
    return combined_features


def demo_model_training(features_df):
    """Demonstrate model training and evaluation"""
    print("\n" + "=" * 50)
    print("DEMO: Model Training and Evaluation")
    print("=" * 50)
    
    # Prepare data
    feature_cols = [col for col in features_df.columns 
                   if col not in ['sample_id', 'condition', 'label', 'noise_level']]
    
    X = features_df[feature_cols].values
    y = features_df['label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model without optimization (for speed in demo)
    print("Training LightGBM model (without optimization for demo speed)...")
    model = OptimizedLightGBM(random_state=42)
    model.fit(X_train, y_train, optimize=False)  # Skip optimization for demo
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (Macro): {f1:.4f}")
    
    # Classification report
    class_names = ['Normal', 'Air Contamination', 'Internal Leakage', 
                   'Valve Sticking', 'Filter Blockage']
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Evaluate with ModelEvaluator
    evaluator = ModelEvaluator(class_names=class_names)
    metrics = evaluator.evaluate_model(model, X_test, y_test, 'LightGBM Demo')
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix('LightGBM Demo')
    plt.savefig('demo_confusion_matrix.png', dpi=150, bbox_inches='tight')
    
    return model, evaluator


def demo_feature_importance(model, feature_names):
    """Demonstrate feature importance analysis"""
    print("\n" + "=" * 50)
    print("DEMO: Feature Importance Analysis")
    print("=" * 50)
    
    # Get feature importance
    importance_df = model.get_feature_importance(feature_names)
    
    print("Top 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Plot feature importance
    top_features = importance_df.head(15)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('demo_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()


def run_complete_demo():
    """Run complete demonstration"""
    print("SHIP STEERING GEAR FAULT DIAGNOSIS - QUICK DEMO")
    print("=" * 60)
    print("This demo showcases the key functionality with a small dataset")
    print("For full experiments, run experiments/main_experiment.py")
    print("=" * 60)
    
    try:
        # Demo 1: Data Simulation
        simulator = demo_data_simulation()
        
        # Demo 2: Feature Extraction
        features_df = demo_feature_extraction()
        
        # Demo 3: Model Training
        model, evaluator = demo_model_training(features_df)
        
        # Demo 4: Feature Importance
        feature_cols = [col for col in features_df.columns 
                       if col not in ['sample_id', 'condition', 'label', 'noise_level']]
        demo_feature_importance(model, feature_cols)
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("Generated files:")
        print("  - demo_signals.png")
        print("  - demo_confusion_matrix.png") 
        print("  - demo_feature_importance.png")
        print("=" * 60)
        print("\nFor the complete research implementation, run:")
        print("python experiments/main_experiment.py")
        
    except Exception as e:
        print(f"Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_complete_demo()
