import numpy as np
import pandas as pd
from pactus import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

class HurricaneDataProcessor:
    """
    Class for processing hurricane trajectory data from the Pactus library.
    Handles data loading, feature extraction, and model training.
    """
    
    def __init__(self, seed=0):
        """Initialize the data processor with optional random seed"""
        self.seed = seed
        self.dataset = None
        self.features_df = None
        self.model = None
        self.category_colors = {
            0: 'blue',
            1: 'green',
            2: 'red',
            3: 'purple',
            4: 'orange',
            5: 'brown'
        }
        
    def load_data(self):
        """Load the hurricane dataset from Pactus"""
        self.dataset = Dataset.hurdat2()
        print(f"Loaded dataset: {self.dataset.name}")
        print(f"Total trajectories: {len(self.dataset.trajs)}")
        print(f"Different classes: {self.dataset.classes}")
        return self.dataset
    
    def get_dataset_summary(self):
        """Get summary statistics of the hurricane dataset"""
        if self.dataset is None:
            self.load_data()
            
        # Calculate trajectory lengths
        traj_lengths = [len(traj) for traj in self.dataset.trajs]
        
        # Calculate trajectory durations
        traj_durations = [(traj.t[-1] - traj.t[0]) for traj in self.dataset.trajs]
        
        # Count trajectories by category
        category_counts = {}
        for i, label in enumerate(self.dataset.labels):
            if label in category_counts:
                category_counts[label] += 1
            else:
                category_counts[label] = 1
                
        # Create summary dictionary
        summary = {
            'dataset_name': self.dataset.name,
            'total_trajectories': len(self.dataset.trajs),
            'classes': len(self.dataset.classes),
            'class_distribution': category_counts,
            'min_trajectory_length': min(traj_lengths),
            'max_trajectory_length': max(traj_lengths),
            'avg_trajectory_length': sum(traj_lengths)/len(traj_lengths),
            'min_duration_hours': min(traj_durations),
            'max_duration_hours': max(traj_durations),
            'avg_duration_hours': sum(traj_durations)/len(traj_durations)
        }
        
        return summary
    
    def extract_features(self):
        """Extract features from hurricane trajectories"""
        if self.dataset is None:
            self.load_data()
            
        features = []
        labels = []
        traj_ids = []
        
        for i, traj in enumerate(self.dataset.trajs):
            # Skip trajectories that are too short
            if len(traj) < 3:
                continue
                
            # Position stats
            r_mean = np.mean(traj.r, axis=0)
            r_std = np.std(traj.r, axis=0)
            
            # Velocity stats - handle short trajectories
            try:
                v_magnitude = np.sqrt(np.sum(traj.v**2, axis=1))
                v_mean = np.mean(v_magnitude)
                v_std = np.std(v_magnitude)
                v_max = np.max(v_magnitude)
            except ValueError:
                # For trajectories too short to estimate velocity
                v_mean = v_std = v_max = 0
            
            # Acceleration stats - handle short trajectories
            try:
                a_magnitude = np.sqrt(np.sum(traj.a**2, axis=1))
                a_mean = np.mean(a_magnitude)
                a_std = np.std(a_magnitude)
                a_max = np.max(a_magnitude)
            except ValueError:
                # For trajectories too short to estimate acceleration
                a_mean = a_std = a_max = 0
            
            # Trajectory length and duration
            traj_length = len(traj)
            traj_duration = traj.t[-1] - traj.t[0]
            
            # Geographical extent
            lon_min, lon_max = traj.bounds[0]
            lat_min, lat_max = traj.bounds[1]
            lon_range = lon_max - lon_min
            lat_range = lat_max - lat_min
            
            # Compile features
            traj_features = {
                'traj_id': traj.traj_id,
                'mean_lon': r_mean[0],
                'mean_lat': r_mean[1],
                'std_lon': r_std[0],
                'std_lat': r_std[1],
                'mean_velocity': v_mean,
                'std_velocity': v_std,
                'max_velocity': v_max,
                'mean_acceleration': a_mean,
                'std_acceleration': a_std,
                'max_acceleration': a_max,
                'traj_length': traj_length,
                'traj_duration': traj_duration,
                'lon_range': lon_range,
                'lat_range': lat_range,
                'category': self.dataset.labels[i]
            }
            
            features.append(traj_features)
            labels.append(self.dataset.labels[i])
            traj_ids.append(traj.traj_id)
            
        # Create DataFrame
        self.features_df = pd.DataFrame(features)
        
        return self.features_df
    
    def train_model(self, test_size=0.3):
        """Train a Random Forest model to predict hurricane category"""
        if self.features_df is None:
            self.extract_features()
            
        # Prepare features and target
        X = self.features_df.drop(['traj_id', 'category'], axis=1)
        y = self.features_df['category']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.seed)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=self.seed)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Get feature importances
        importances = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'model': self.model,
            'report': report,
            'feature_importance': feature_importance,
            'confusion_matrix': cm,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def save_model(self, filepath='hurricane_model.pkl'):
        """Save the trained model to a file"""
        if self.model is None:
            print("No model to save. Please train a model first.")
            return False
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"Model saved to {filepath}")
        return True
    
    def load_model(self, filepath='hurricane_model.pkl'):
        """Load a trained model from a file"""
        if not os.path.exists(filepath):
            print(f"Model file {filepath} not found.")
            return False
        
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"Model loaded from {filepath}")
        return True
    
    def predict_category(self, features):
        """Predict hurricane category from features"""
        if self.model is None:
            print("No model available. Please train or load a model first.")
            return None
        
        # Ensure features have the correct format
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        # Make prediction
        prediction = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        
        return {
            'prediction': prediction[0],
            'probabilities': probabilities[0]
        }
    
    def get_trajectory_by_id(self, traj_id):
        """Get a trajectory by its ID"""
        if self.dataset is None:
            self.load_data()
            
        for traj in self.dataset.trajs:
            if traj.traj_id == traj_id:
                return traj
        
        return None
    
    def normalize_trajectory(self, traj):
        """Normalize trajectory to start at origin and have unit length"""
        # Shift to start at origin
        r_normalized = traj.r - traj.r[0]
        # Scale to unit length (max distance from origin)
        max_dist = np.max(np.sqrt(np.sum(r_normalized**2, axis=1)))
        if max_dist > 0:
            r_normalized = r_normalized / max_dist
        return r_normalized
    
    def get_category_color(self, category):
        """Get the color associated with a hurricane category"""
        return self.category_colors.get(category, 'gray')
    
    def get_sample_trajectories(self, n_per_category=5):
        """Get sample trajectories for each category"""
        if self.dataset is None:
            self.load_data()
            
        samples = {}
        
        for category in self.dataset.classes:
            # Get indices of trajectories with this category
            category_indices = [i for i, label in enumerate(self.dataset.labels) 
                               if label == category]
            
            # Sample trajectories
            sample_size = min(n_per_category, len(category_indices))
            if sample_size > 0:
                sampled_indices = np.random.choice(category_indices, sample_size, replace=False)
                samples[category] = [self.dataset.trajs[i] for i in sampled_indices]
            else:
                samples[category] = []
        
        return samples

# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = HurricaneDataProcessor()
    
    # Load data
    processor.load_data()
    
    # Get dataset summary
    summary = processor.get_dataset_summary()
    print("\nDataset Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Extract features
    features_df = processor.extract_features()
    print("\nFeatures DataFrame:")
    print(features_df.head())
    
    # Train model
    model_results = processor.train_model()
    print("\nModel Training Results:")
    print(f"Accuracy: {model_results['report']['accuracy']:.4f}")
    
    # Print feature importance
    print("\nFeature Importance:")
    print(model_results['feature_importance'].head(10))
    
    # Save model
    processor.save_model()
    
    print("\nData processing complete.")
