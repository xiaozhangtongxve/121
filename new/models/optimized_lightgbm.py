"""
Optimized LightGBM Model for Ship Steering Gear Fault Diagnosis
Implements LightGBM with hyperparameter optimization using PSO and GWO
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class OptimizedLightGBM:
    """LightGBM classifier with hyperparameter optimization"""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize optimized LightGBM model
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_params = None
        self.optimization_history = []
        
        # Default parameter ranges for optimization
        self.param_ranges = {
            'n_estimators': (50, 500),
            'learning_rate': (0.01, 0.3),
            'num_leaves': (10, 100),
            'max_depth': (3, 15),
            'reg_alpha': (0.0, 1.0),
            'reg_lambda': (0.0, 1.0),
            'min_child_samples': (5, 50),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0)
        }
    
    def objective_function(self, params: Dict, X: np.ndarray, y: np.ndarray, 
                          cv_folds: int = 5, scoring: str = 'f1_macro') -> float:
        """
        Objective function for hyperparameter optimization
        
        Args:
            params: Dictionary of hyperparameters
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Negative cross-validation score (for minimization)
        """
        try:
            # Convert continuous parameters to appropriate types
            lgb_params = {
                'n_estimators': int(params['n_estimators']),
                'learning_rate': params['learning_rate'],
                'num_leaves': int(params['num_leaves']),
                'max_depth': int(params['max_depth']),
                'reg_alpha': params['reg_alpha'],
                'reg_lambda': params['reg_lambda'],
                'min_child_samples': int(params['min_child_samples']),
                'subsample': params['subsample'],
                'colsample_bytree': params['colsample_bytree'],
                'random_state': self.random_state,
                'verbose': -1,
                'n_jobs': -1
            }
            
            # Create model
            model = lgb.LGBMClassifier(**lgb_params)
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=cv_folds, 
                                                                       shuffle=True, 
                                                                       random_state=self.random_state),
                                      scoring=scoring, n_jobs=-1)
            
            # Return negative score for minimization
            return -np.mean(cv_scores)
            
        except Exception as e:
            # Return large penalty for invalid parameters
            return 1.0
    
    def particle_swarm_optimization(self, X: np.ndarray, y: np.ndarray, 
                                  n_particles: int = 20, n_iterations: int = 50,
                                  w: float = 0.7, c1: float = 1.5, c2: float = 1.5) -> Dict:
        """
        Particle Swarm Optimization for hyperparameter tuning
        
        Args:
            X: Feature matrix
            y: Target vector
            n_particles: Number of particles
            n_iterations: Number of iterations
            w: Inertia weight
            c1: Cognitive parameter
            c2: Social parameter
            
        Returns:
            Best parameters found
        """
        print("Starting Particle Swarm Optimization...")
        
        # Initialize particles
        particles = []
        velocities = []
        personal_best = []
        personal_best_scores = []
        
        # Parameter bounds
        bounds = np.array([[self.param_ranges[param][0], self.param_ranges[param][1]] 
                          for param in self.param_ranges.keys()])
        
        # Initialize particles randomly
        for _ in range(n_particles):
            particle = {}
            velocity = {}
            for i, param in enumerate(self.param_ranges.keys()):
                low, high = bounds[i]
                particle[param] = np.random.uniform(low, high)
                velocity[param] = np.random.uniform(-0.1 * (high - low), 0.1 * (high - low))
            
            particles.append(particle)
            velocities.append(velocity)
            personal_best.append(particle.copy())
            personal_best_scores.append(float('inf'))
        
        # Global best
        global_best = None
        global_best_score = float('inf')
        
        # Optimization loop
        for iteration in range(n_iterations):
            for i in range(n_particles):
                # Evaluate particle
                score = self.objective_function(particles[i], X, y)
                
                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best[i] = particles[i].copy()
                
                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best = particles[i].copy()
            
            # Update particles
            for i in range(n_particles):
                for param in self.param_ranges.keys():
                    # Update velocity
                    r1, r2 = np.random.random(), np.random.random()
                    velocities[i][param] = (w * velocities[i][param] + 
                                          c1 * r1 * (personal_best[i][param] - particles[i][param]) +
                                          c2 * r2 * (global_best[param] - particles[i][param]))
                    
                    # Update position
                    particles[i][param] += velocities[i][param]
                    
                    # Apply bounds
                    low, high = self.param_ranges[param]
                    particles[i][param] = np.clip(particles[i][param], low, high)
            
            # Store optimization history
            self.optimization_history.append({
                'iteration': iteration,
                'best_score': -global_best_score,
                'algorithm': 'PSO'
            })
            
            if iteration % 10 == 0:
                print(f"PSO Iteration {iteration}: Best Score = {-global_best_score:.4f}")
        
        print(f"PSO completed. Best Score: {-global_best_score:.4f}")
        return global_best
    
    def grey_wolf_optimization(self, X: np.ndarray, y: np.ndarray,
                             n_wolves: int = 20, n_iterations: int = 50) -> Dict:
        """
        Grey Wolf Optimization for hyperparameter tuning
        
        Args:
            X: Feature matrix
            y: Target vector
            n_wolves: Number of wolves
            n_iterations: Number of iterations
            
        Returns:
            Best parameters found
        """
        print("Starting Grey Wolf Optimization...")
        
        # Parameter bounds
        bounds = np.array([[self.param_ranges[param][0], self.param_ranges[param][1]] 
                          for param in self.param_ranges.keys()])
        param_names = list(self.param_ranges.keys())
        
        # Initialize wolves
        wolves = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_wolves, len(param_names)))
        
        # Initialize alpha, beta, delta wolves
        alpha_pos = np.zeros(len(param_names))
        beta_pos = np.zeros(len(param_names))
        delta_pos = np.zeros(len(param_names))
        
        alpha_score = float('inf')
        beta_score = float('inf')
        delta_score = float('inf')
        
        # Optimization loop
        for iteration in range(n_iterations):
            for i in range(n_wolves):
                # Convert position to parameter dictionary
                params = {param_names[j]: wolves[i, j] for j in range(len(param_names))}
                
                # Evaluate wolf
                score = self.objective_function(params, X, y)
                
                # Update alpha, beta, delta
                if score < alpha_score:
                    delta_score = beta_score
                    delta_pos = beta_pos.copy()
                    beta_score = alpha_score
                    beta_pos = alpha_pos.copy()
                    alpha_score = score
                    alpha_pos = wolves[i].copy()
                elif score < beta_score:
                    delta_score = beta_score
                    delta_pos = beta_pos.copy()
                    beta_score = score
                    beta_pos = wolves[i].copy()
                elif score < delta_score:
                    delta_score = score
                    delta_pos = wolves[i].copy()
            
            # Update wolf positions
            a = 2 - iteration * (2 / n_iterations)  # Linearly decreasing from 2 to 0
            
            for i in range(n_wolves):
                for j in range(len(param_names)):
                    # Alpha wolf influence
                    r1, r2 = np.random.random(), np.random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha_pos[j] - wolves[i, j])
                    X1 = alpha_pos[j] - A1 * D_alpha
                    
                    # Beta wolf influence
                    r1, r2 = np.random.random(), np.random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta_pos[j] - wolves[i, j])
                    X2 = beta_pos[j] - A2 * D_beta
                    
                    # Delta wolf influence
                    r1, r2 = np.random.random(), np.random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta_pos[j] - wolves[i, j])
                    X3 = delta_pos[j] - A3 * D_delta
                    
                    # Update position
                    wolves[i, j] = (X1 + X2 + X3) / 3
                    
                    # Apply bounds
                    wolves[i, j] = np.clip(wolves[i, j], bounds[j, 0], bounds[j, 1])
            
            # Store optimization history
            self.optimization_history.append({
                'iteration': iteration,
                'best_score': -alpha_score,
                'algorithm': 'GWO'
            })
            
            if iteration % 10 == 0:
                print(f"GWO Iteration {iteration}: Best Score = {-alpha_score:.4f}")
        
        # Convert best position to parameter dictionary
        best_params = {param_names[j]: alpha_pos[j] for j in range(len(param_names))}
        
        print(f"GWO completed. Best Score: {-alpha_score:.4f}")
        return best_params
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, 
                                method: str = 'PSO', **kwargs) -> Dict:
        """
        Optimize hyperparameters using specified method
        
        Args:
            X: Feature matrix
            y: Target vector
            method: Optimization method ('PSO' or 'GWO')
            **kwargs: Additional arguments for optimization method
            
        Returns:
            Best parameters found
        """
        self.optimization_history = []
        
        if method.upper() == 'PSO':
            self.best_params = self.particle_swarm_optimization(X, y, **kwargs)
        elif method.upper() == 'GWO':
            self.best_params = self.grey_wolf_optimization(X, y, **kwargs)
        else:
            raise ValueError("Method must be 'PSO' or 'GWO'")
        
        return self.best_params
    
    def fit(self, X: np.ndarray, y: np.ndarray, optimize: bool = True, 
            optimization_method: str = 'PSO') -> None:
        """
        Fit the LightGBM model
        
        Args:
            X: Feature matrix
            y: Target vector
            optimize: Whether to optimize hyperparameters
            optimization_method: Method for optimization ('PSO' or 'GWO')
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if optimize:
            # Optimize hyperparameters
            best_params = self.optimize_hyperparameters(X_scaled, y_encoded, 
                                                       method=optimization_method)
            
            # Convert parameters to appropriate types
            lgb_params = {
                'n_estimators': int(best_params['n_estimators']),
                'learning_rate': best_params['learning_rate'],
                'num_leaves': int(best_params['num_leaves']),
                'max_depth': int(best_params['max_depth']),
                'reg_alpha': best_params['reg_alpha'],
                'reg_lambda': best_params['reg_lambda'],
                'min_child_samples': int(best_params['min_child_samples']),
                'subsample': best_params['subsample'],
                'colsample_bytree': best_params['colsample_bytree'],
                'random_state': self.random_state,
                'verbose': -1
            }
        else:
            # Use default parameters
            lgb_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'max_depth': -1,
                'random_state': self.random_state,
                'verbose': -1
            }
        
        # Train final model
        self.model = lgb.LGBMClassifier(**lgb_params)
        self.model.fit(X_scaled, y_encoded)
        
        print("Model training completed.")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        y_pred_encoded = self.model.predict(X_scaled)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def plot_optimization_convergence(self, save_path: Optional[str] = None) -> None:
        """Plot optimization convergence"""
        if not self.optimization_history:
            print("No optimization history available.")
            return
        
        history_df = pd.DataFrame(self.optimization_history)
        
        plt.figure(figsize=(10, 6))
        plt.plot(history_df['iteration'], history_df['best_score'], 'b-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Best Score')
        plt.title(f'{history_df["algorithm"].iloc[0]} Optimization Convergence')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Get feature importance from trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        importance = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
