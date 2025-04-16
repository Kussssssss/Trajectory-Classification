import numpy as np
import pandas as pd
from pactus import Dataset
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
    
    def haversine(self, lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points
        using the Haversine formula
        """
        R = 6371  # bán kính trái đất (km)
        dlon = np.radians(lon2 - lon1)
        dlat = np.radians(lat2 - lat1)
        a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c
    
    def extract_features(self):
        """
        Extract comprehensive features from hurricane trajectories
        with advanced spatial, temporal, and kinematic features
        """
        if self.dataset is None:
            self.load_data()
            
        features_list = []
        
        for i, traj in enumerate(self.dataset.trajs):
            feat = {}
            # Đặc trưng cơ bản từ quỹ đạo
            feat['traj_id'] = traj.traj_id
            feat['t_0'] = traj.t_0
            feat['uniformly_spaced'] = 1 if traj.uniformly_spaced else 0
            feat['min_lon'] = traj.bounds[0][0]
            feat['max_lon'] = traj.bounds[0][1]
            feat['min_lat'] = traj.bounds[1][0]
            feat['max_lat'] = traj.bounds[1][1]
            feat['dim'] = traj.dim
            feat['points_count'] = len(traj.r)  # Số điểm quan sát
            
            # Lưu nhãn
            if hasattr(self.dataset, 'labels'):
                feat['class'] = self.dataset.labels[i]
            
            # Đặc trưng không gian
            r = traj.r  # tọa độ [lon, lat]
            lon = r[:, 0]
            lat = r[:, 1]
            
            # Thống kê cơ bản về tọa độ
            feat['mean_lon'] = np.mean(lon)
            feat['mean_lat'] = np.mean(lat)
            feat['median_lon'] = np.median(lon)
            feat['median_lat'] = np.median(lat)
            feat['std_lon'] = np.std(lon)
            feat['std_lat'] = np.std(lat)
            feat['lon_range'] = np.ptp(lon)
            feat['lat_range'] = np.ptp(lat)
            feat['start_lon'] = lon[0]
            feat['start_lat'] = lat[0]
            feat['end_lon'] = lon[-1]
            feat['end_lat'] = lat[-1]
            
            # Tính diện tích và chu vi hình chữ nhật bao quanh
            feat['bbox_area'] = feat['lon_range'] * feat['lat_range']
            feat['bbox_perimeter'] = 2 * (feat['lon_range'] + feat['lat_range'])
            
            # Tính toán tâm hình học
            feat['centroid_lon'] = (feat['min_lon'] + feat['max_lon']) / 2
            feat['centroid_lat'] = (feat['min_lat'] + feat['max_lat']) / 2
            
            # Các điểm phân vị (để nắm bắt hình dạng)
            if len(r) >= 4:
                q1_idx = len(r) // 4
                q2_idx = len(r) // 2
                q3_idx = 3 * len(r) // 4
                
                feat['q1_lon'] = lon[q1_idx]
                feat['q1_lat'] = lat[q1_idx]
                feat['q2_lon'] = lon[q2_idx]
                feat['q2_lat'] = lat[q2_idx]
                feat['q3_lon'] = lon[q3_idx]
                feat['q3_lat'] = lat[q3_idx]
            else:
                # Với quỹ đạo quá ngắn
                feat['q1_lon'] = feat['q2_lon'] = feat['q3_lon'] = feat['mean_lon']
                feat['q1_lat'] = feat['q2_lat'] = feat['q3_lat'] = feat['mean_lat']
            
            # Tính tổng khoảng cách di chuyển theo Haversine
            path_length = 0
            for j in range(1, len(r)):
                path_length += self.haversine(lon[j-1], lat[j-1], lon[j], lat[j])
            feat['path_length'] = path_length
            
            # Khoảng cách trực tiếp (đường thẳng từ điểm đầu đến điểm cuối)
            direct_distance = self.haversine(lon[0], lat[0], lon[-1], lat[-1])
            feat['direct_distance'] = direct_distance
            
            # Độ uốn cong (tỷ lệ giữa tổng đường đi và khoảng cách trực tiếp)
            feat['sinuosity'] = path_length / max(direct_distance, 0.001)
            
            # Hiệu quả di chuyển (tỷ lệ giữa khoảng cách trực tiếp và tổng đường đi)
            feat['displacement_efficiency'] = direct_distance / max(path_length, 0.001)
            
            # Đặc trưng thời gian
            t = traj.t
            feat['traj_duration'] = t[-1] - t[0]
            feat['duration_hours'] = feat['traj_duration'] / 3600
            feat['duration_days'] = feat['traj_duration'] / (3600 * 24)
            
            # Khoảng thời gian giữa các quan sát
            dt = np.diff(t)
            feat['avg_time_between_points'] = np.mean(dt)
            feat['median_time_between_points'] = np.median(dt)
            feat['std_time_between_points'] = np.std(dt)
            feat['min_time_between_points'] = np.min(dt) if len(dt) > 0 else 0
            feat['max_time_between_points'] = np.max(dt) if len(dt) > 0 else 0
            
            # Đặc trưng vận tốc
            v = traj.v
            v_norm = np.linalg.norm(v, axis=1)
            feat['mean_velocity'] = np.mean(v_norm)
            feat['median_velocity'] = np.median(v_norm)
            feat['std_velocity'] = np.std(v_norm)
            feat['max_velocity'] = np.max(v_norm)
            feat['min_velocity'] = np.min(v_norm)
            feat['velocity_range'] = feat['max_velocity'] - feat['min_velocity']
            
            # Tính phân vị của vận tốc
            feat['velocity_25pct'] = np.percentile(v_norm, 25)
            feat['velocity_75pct'] = np.percentile(v_norm, 75)
            feat['velocity_iqr'] = feat['velocity_75pct'] - feat['velocity_25pct']
            
            # Thống kê thành phần vận tốc
            feat['mean_velocity_lon'] = np.mean(v[:, 0])
            feat['mean_velocity_lat'] = np.mean(v[:, 1])
            feat['std_velocity_lon'] = np.std(v[:, 0])
            feat['std_velocity_lat'] = np.std(v[:, 1])
            
            # Đếm số lần thay đổi hướng vận tốc
            v_changes = 0
            for j in range(1, len(v) - 1):
                cross1 = v[j-1, 0] * v[j, 1] - v[j-1, 1] * v[j, 0]
                cross2 = v[j, 0] * v[j+1, 1] - v[j, 1] * v[j+1, 0]
                if np.sign(cross1) != np.sign(cross2):
                    v_changes += 1
            feat['v_direction_changes'] = v_changes
            feat['v_direction_changes_ratio'] = v_changes / max(len(v) - 2, 1)
            
            # Đặc trưng gia tốc
            try:
                a = traj.a
                valid_accel = True
            except (ValueError, IndexError, AttributeError):
                a = np.zeros((len(traj.t), traj.r.shape[1]))
                valid_accel = False
            
            if valid_accel and len(a) > 0:
                a_norm = np.linalg.norm(a, axis=1)
                epsilon = 1e-8
                a_norm[a_norm < epsilon] = 0
                
                feat['mean_acceleration'] = np.mean(a_norm)
                feat['median_acceleration'] = np.median(a_norm)
                feat['std_acceleration'] = np.std(a_norm)
                feat['max_acceleration'] = np.max(a_norm)
                
                # Thành phần gia tốc
                feat['mean_acceleration_lon'] = np.mean(a[:, 0])
                feat['mean_acceleration_lat'] = np.mean(a[:, 1])
                
                # Đếm số lần thay đổi hướng gia tốc
                a_changes = 0
                for j in range(1, len(a) - 1):
                    if (np.sign(a[j, 0]) != np.sign(a[j-1, 0])) or (np.sign(a[j, 1]) != np.sign(a[j-1, 1])):
                        a_changes += 1
                feat['a_direction_changes'] = a_changes
                feat['a_direction_changes_ratio'] = a_changes / max(len(a) - 2, 1)
            else:
                # Giá trị mặc định nếu không tính được gia tốc
                feat['mean_acceleration'] = feat['median_acceleration'] = feat['std_acceleration'] = 0
                feat['max_acceleration'] = 0
                feat['mean_acceleration_lon'] = feat['mean_acceleration_lat'] = 0
                feat['a_direction_changes'] = feat['a_direction_changes_ratio'] = 0
            
            # Đặc trưng về độ cong
            curvature = []
            for j in range(1, len(r) - 1):
                vec1 = r[j] - r[j-1]
                vec2 = r[j+1] - r[j]
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                if norm1 > 0 and norm2 > 0:
                    cos_angle = np.clip(dot_product / (norm1 * norm2), -1, 1)
                    angle = np.arccos(cos_angle)
                    curvature.append(angle)
            
            if curvature:
                feat['curvature_mean'] = np.mean(curvature)
                feat['curvature_median'] = np.median(curvature)
                feat['curvature_std'] = np.std(curvature)
                feat['curvature_max'] = np.max(curvature)
                
                # Số lần rẽ gấp (điểm có độ cong cao)
                sharp_turn_threshold = np.radians(45)  # Xác định rẽ gấp là >= 45 độ
                feat['sharp_turns_count'] = sum(c > sharp_turn_threshold for c in curvature)
                feat['sharp_turns_ratio'] = feat['sharp_turns_count'] / max(len(curvature), 1)
            else:
                feat['curvature_mean'] = feat['curvature_median'] = feat['curvature_std'] = 0
                feat['curvature_max'] = 0
                feat['sharp_turns_count'] = feat['sharp_turns_ratio'] = 0
            
            features_list.append(feat)
        
        self.features_df = pd.DataFrame(features_list)
        return self.features_df
    
    def create_interaction_features(self, df=None):
        """
        Tạo các đặc trưng tương tác giữa các đặc trưng quan trọng
        """
        if df is None:
            if self.features_df is None:
                self.extract_features()
            df = self.features_df
            
        # Chọn các đặc trưng quan trọng nhất cho tương tác
        important_features = ['mean_velocity', 'path_length', 'curvature_mean', 
                             'sinuosity', 'max_velocity', 'duration_hours']
        
        result_df = df.copy()
        
        # Tạo đặc trưng tỷ lệ
        for i in range(len(important_features)):
            for j in range(i+1, len(important_features)):
                col1, col2 = important_features[i], important_features[j]
                if col1 in df.columns and col2 in df.columns:
                    # Đặc trưng tỷ lệ (với epsilon nhỏ để tránh chia cho 0)
                    feature_name = f'{col1}_div_{col2}'
                    result_df[feature_name] = df[col1] / (df[col2] + 1e-10)
                    
                    # Đặc trưng tích
                    feature_name = f'{col1}_mul_{col2}'
                    result_df[feature_name] = df[col1] * df[col2]
        
        self.features_df = result_df
        return result_df
    
    def handle_outliers(self, df=None, method='winsorize'):
        """
        Xử lý outlier trong các cột số với nhiều phương pháp khác nhau
        """
        if df is None:
            if self.features_df is None:
                self.extract_features()
            df = self.features_df
            
        # Lấy tất cả cột số trừ ID và class
        cols = df.select_dtypes(include=np.number).columns.difference(['traj_id', 'class'])
        
        result_df = df.copy()
        
        for col in cols:
            if method == 'winsorize':
                # Winsorization: giới hạn ở phân vị 1% và 99%
                q1, q99 = np.percentile(result_df[col].dropna(), [1, 99])
                result_df[col] = result_df[col].clip(q1, q99)
            
            elif method == 'iqr':
                # Phương pháp IQR
                q1, q3 = np.percentile(result_df[col].dropna(), [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                result_df[col] = result_df[col].clip(lower_bound, upper_bound)
            
            elif method == 'zscore':
                # Phương pháp Z-score
                mean = result_df[col].mean()
                std = result_df[col].std()
                if std > 0:  # Tránh chia cho 0
                    z_scores = (result_df[col] - mean) / std
                    result_df[col] = np.where(np.abs(z_scores) > 3, mean, result_df[col])
        
        self.features_df = result_df
        return result_df
    
    def extract_features(self):
        """Extract basic features from hurricane trajectories"""
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
    
    def process_data_pipeline(self, outlier_method='winsorize', create_interactions=True):
        """
        Run the complete data processing pipeline:
        1. Extract enhanced features
        2. Handle outliers
        3. Create interaction features (optional)
        """
        # Step 1: Extract enhanced features
        self.extract_features()
        
        # Step 2: Handle outliers
        self.handle_outliers(method=outlier_method)
        
        # Step 3: Create interaction features if requested
        if create_interactions:
            self.create_interaction_features()
            
        return self.features_df
    
    def train_model(self, test_size=0.3, use_features=True):
        """Train a Random Forest model to predict hurricane category"""
        if self.features_df is None:
            if use_features:
                self.process_data_pipeline()
            else:
                self.extract_features()
            
        # Prepare features and target
        X = self.features_df.drop(['traj_id', 'class' if 'class' in self.features_df.columns else 'category'], axis=1)
        y = self.features_df['class'] if 'class' in self.features_df.columns else self.features_df['category']
        
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