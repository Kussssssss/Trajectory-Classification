from pactus import Dataset, featurizers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from yupi.graphics import plot_2d, plot_hist
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.lines as mlines
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Load the dataset
SEED = 0
datasets = Dataset.hurdat2()

universal_features = featurizers.UniversalFeaturizer()
# universal_features.describe(detailed=True)
# Explore available featurizers
print("Available featurizers in Pactus:")
for featurizer_name in dir(featurizers):
    if not featurizer_name.startswith('_'):
        print(f"- {featurizer_name}")

# Examine the first featurizer to understand its methods
velocity_feat = featurizers.VelocityFeaturizer()
print("\nExamining VelocityFeaturizer methods:")
for method_name in dir(velocity_feat):
    if not method_name.startswith('_'):
        print(f"- {method_name}")

# Extract basic statistics from trajectories
def extract_basic_stats(trajs, labels):
    """Extract basic statistics from trajectories"""
    stats = []
    
    for traj in trajs:
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
        
        # Compile stats
        traj_stats = [
            r_mean[0], r_mean[1],  # Mean longitude and latitude
            r_std[0], r_std[1],    # Std of longitude and latitude
            v_mean, v_std, v_max,  # Velocity statistics
            a_mean, a_std, a_max,  # Acceleration statistics
            traj_length, traj_duration,  # Length and duration
            lon_range, lat_range   # Geographical extent
        ]
        
        stats.append(traj_stats)
    
    # Create feature names
    feature_names = [
        'mean_lon', 'mean_lat',
        'std_lon', 'std_lat',
        'mean_velocity', 'std_velocity', 'max_velocity',
        'mean_acceleration', 'std_acceleration', 'max_acceleration',
        'traj_length', 'traj_duration',
        'lon_range', 'lat_range'
    ]
    
    return np.array(stats), feature_names
feature_names = [
    "time_mean",
    "time_median",
    "time_kurtosis",
    "time_autocorr",
    "time_min",
    "time_max",
    "time_range",
    "time_std",
    "time_var",
    "time_coeff_var",
    "time_iqr",
    "time_jump_mean",
    "time_jump_median",
    "time_jump_kurtosis",
    "time_jump_autocorr",
    "time_jump_min",
    "time_jump_max",
    "time_jump_range",
    "time_jump_std",
    "time_jump_var",
    "time_jump_coeff_var",
    "time_jump_iqr",
    "distance",
    "displacement",
    "jump_mean",
    "jump_median",
    "jump_kurtosis",
    "jump_autocorr",
    "jump_min",
    "jump_max",
    "jump_range",
    "jump_std",
    "jump_var",
    "jump_coeff_var",
    "jump_iqr",
    "velocity_mean",
    "velocity_median",
    "velocity_kurtosis",
    "velocity_autocorr",
    "velocity_min",
    "velocity_max",
    "velocity_range",
    "velocity_std",
    "velocity_var",
    "velocity_coeff_var",
    "velocity_iqr",
    "velocity_stop_rate",
    "velocity_change_rate",
    "acceleration_mean",
    "acceleration_median",
    "acceleration_kurtosis",
    "acceleration_autocorr",
    "acceleration_min",
    "acceleration_max",
    "acceleration_range",
    "acceleration_std",
    "acceleration_var",
    "acceleration_coeff_var",
    "acceleration_iqr",
    "acceleration_change_rate_mean",
    "acceleration_change_rate_median",
    "acceleration_change_rate_kurtosis",
    "acceleration_change_rate_autocorr",
    "acceleration_change_rate_min",
    "acceleration_change_rate_max",
    "acceleration_change_rate_range",
    "acceleration_change_rate_std",
    "acceleration_change_rate_var",
    "acceleration_change_rate_coeff_var",
    "acceleration_change_rate_iqr",
    "angle_mean",
    "angle_median",
    "angle_kurtosis",
    "angle_autocorr",
    "angle_min",
    "angle_max",
    "angle_range",
    "angle_std",
    "angle_var",
    "angle_coeff_var",
    "angle_iqr",
    "turning_angle_mean",
    "turning_angle_median",
    "turning_angle_kurtosis",
    "turning_angle_autocorr",
    "turning_angle_min",
    "turning_angle_max",
    "turning_angle_range",
    "turning_angle_std",
    "turning_angle_var",
    "turning_angle_coeff_var",
    "turning_angle_iqr",
    "turning_angle_change_rate_mean",
    "turning_angle_change_rate_median",
    "turning_angle_change_rate_kurtosis",
    "turning_angle_change_rate_autocorr",
    "turning_angle_change_rate_min",
    "turning_angle_change_rate_max",
    "turning_angle_change_rate_range",
    "turning_angle_change_rate_std",
    "turning_angle_change_rate_var",
    "turning_angle_change_rate_coeff_var",
    "turning_angle_change_rate_iqr"
]
# Extract features
filtered_trajs, filtered_labels = [traj for traj, label in zip(datasets.trajs, datasets.labels) 
                  if len(traj) >= 5 
                  and traj.r.delta.norm.sum() > 0], [label for traj, label in zip(datasets.trajs, datasets.labels) 
                  if len(traj) >= 5 
                  and traj.r.delta.norm.sum() > 0]
print(f"Số lượng quỹ đạo hợp lệ: {len(filtered_trajs)} / {len(datasets.trajs)}")

X = extract_basic_stats(filtered_trajs, filtered_labels)
y = np.array(datasets.labels)

print(f"\nExtracted basic statistics:")
print(f"Number of features: {len(feature_names)}")
print(f"Feature names: {feature_names}")
print(f"Feature matrix shape: {X.shape}")

# Create a DataFrame for easier analysis
df = pd.DataFrame(X, columns=feature_names)
df['hurricane_category'] = y

print("\nSample of extracted features:")
print(df.head())
df.to_csv("hurricane_features.csv", index=False)
# Analyze feature importance for hurricane category prediction
# from sklearn.ensemble import RandomForestClassifier
from pactus.models.random_forest_model import RandomForestModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Function to analyze feature importance
def analyze_feature_importance(X, y, feature_names, title):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)
    
    # Train a random forest classifier
    rf = RandomForestModel(n_estimators=100, random_state=SEED, featurizer=universal_features)
    rf.fit(X_train, y_train)
    
    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Print feature ranking
    print(f"\n{title} - Feature ranking:")
    for i in range(min(10, len(feature_names))):
        print(f"{i+1}. {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")
    
    # Plot feature importances
    plt.figure(figsize=(12, 6))
    plt.title(f"Feature Importances - {title}")
    plt.bar(range(min(20, len(feature_names))), 
            importances[indices[:20]], 
            align="center")
    plt.xticks(range(min(20, len(feature_names))), 
              [feature_names[i] for i in indices[:20]], 
              rotation=90)
    plt.tight_layout()
    plt.savefig(f'feature_importance_{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    
    # Evaluate model performance
    y_pred = rf.predict(X_test)
    print(f"\nClassification Report - {title}:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(set(y)), 
                yticklabels=sorted(set(y)))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {title}')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    
    return rf, importances, indices

# Analyze feature importance
rf, importances, indices = analyze_feature_importance(
    X, y, feature_names, "Hurricane Features")

# Visualize hurricane trajectories by velocity and acceleration
# Create a custom colormap for hurricane categories
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
cmap = LinearSegmentedColormap.from_list('hurricane_categories', colors, N=6)

# Sample trajectories for visualization
sample_size = 50
sample_indices = np.random.choice(len(datasets.trajs), sample_size, replace=False)

# Create subplots for trajectory visualization
plt.figure(figsize=(20, 16))

# Plot 1: Trajectories on map
ax1 = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cfeature.BORDERS, linestyle=':')
ax1.add_feature(cfeature.LAND, facecolor='lightgray')
ax1.add_feature(cfeature.OCEAN, facecolor='lightblue')

for i in sample_indices:
    traj = datasets.trajs[i]
    category = datasets.labels[i]
    lons = traj.r[:, 0]
    lats = traj.r[:, 1]
    ax1.plot(lons, lats, color=colors[category], alpha=0.7, 
             linewidth=1.5, transform=ccrs.PlateCarree())
    ax1.scatter(lons[0], lats[0], color=colors[category], 
               marker='o', s=30, transform=ccrs.PlateCarree())

# Add legend
legend_handles = []
for i, color in enumerate(colors):
    handle = mlines.Line2D([], [], color=color, marker='o', 
                          markersize=5, label=f'Category {i}')
    legend_handles.append(handle)
ax1.legend(handles=legend_handles, title='Hurricane Categories')
ax1.set_title('Hurricane Trajectories by Category')

# Plot 2: Velocity distribution
ax2 = plt.subplot(2, 2, 2)
velocity_means = []
categories = []

for i in range(len(datasets.trajs)):
    traj = datasets.trajs[i]
    try:
        v_magnitude = np.sqrt(np.sum(traj.v**2, axis=1))
        velocity_means.append(np.mean(v_magnitude))
        categories.append(datasets.labels[i])
    except ValueError:
        # Skip trajectories too short to estimate velocity
        continue

velocity_df = pd.DataFrame({
    'Mean Velocity': velocity_means,
    'Category': categories
})

sns.boxplot(x='Category', y='Mean Velocity', data=velocity_df, ax=ax2, palette=colors)
ax2.set_title('Hurricane Velocity Distribution by Category')
ax2.set_xlabel('Hurricane Category')
ax2.set_ylabel('Mean Velocity (distance/time)')

# Plot 3: Trajectory length distribution
ax3 = plt.subplot(2, 2, 3)
traj_lengths = []
categories = []

for i in range(len(datasets.trajs)):
    traj = datasets.trajs[i]
    traj_lengths.append(len(traj))
    categories.append(datasets.labels[i])

length_df = pd.DataFrame({
    'Trajectory Length': traj_lengths,
    'Category': categories
})

sns.boxplot(x='Category', y='Trajectory Length', data=length_df, ax=ax3, palette=colors)
ax3.set_title('Hurricane Trajectory Length Distribution by Category')
ax3.set_xlabel('Hurricane Category')
ax3.set_ylabel('Number of Points')

# Plot 4: Geographical extent distribution
ax4 = plt.subplot(2, 2, 4)
geo_extents = []
categories = []

for i in range(len(datasets.trajs)):
    traj = datasets.trajs[i]
    lon_min, lon_max = traj.bounds[0]
    lat_min, lat_max = traj.bounds[1]
    geo_extent = (lon_max - lon_min) * (lat_max - lat_min)  # Area covered
    geo_extents.append(geo_extent)
    categories.append(datasets.labels[i])

extent_df = pd.DataFrame({
    'Geographical Extent': geo_extents,
    'Category': categories
})

sns.boxplot(x='Category', y='Geographical Extent', data=extent_df, ax=ax4, palette=colors)
ax4.set_title('Hurricane Geographical Extent by Category')
ax4.set_xlabel('Hurricane Category')
ax4.set_ylabel('Area Covered (degrees²)')

plt.tight_layout()
plt.savefig('hurricane_analysis.png', dpi=300, bbox_inches='tight')

# Create a correlation matrix of features
plt.figure(figsize=(16, 14))
# Calculate correlation matrix
corr_matrix = df.corr()

# Plot correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Hurricane Features')
plt.tight_layout()
plt.savefig('feature_correlation.png', dpi=300, bbox_inches='tight')

# Analyze trajectory patterns by category
plt.figure(figsize=(15, 10))

# Create a function to normalize trajectories for comparison
def normalize_trajectory(traj):
    """Normalize trajectory to start at origin and have unit length"""
    # Shift to start at origin
    r_normalized = traj.r - traj.r[0]
    # Scale to unit length (max distance from origin)
    max_dist = np.max(np.sqrt(np.sum(r_normalized**2, axis=1)))
    if max_dist > 0:
        r_normalized = r_normalized / max_dist
    return r_normalized

# Plot normalized trajectories by category
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for category in range(6):
    ax = axes[category]
    # Get trajectories for this category
    category_indices = [i for i, label in enumerate(datasets.labels) if label == category]
    # Sample trajectories
    sample_size = min(20, len(category_indices))
    sampled_indices = np.random.choice(category_indices, sample_size, replace=False)
    
    for idx in sampled_indices:
        traj = datasets.trajs[idx]
        # Only plot trajectories with at least 3 points
        if len(traj) >= 3:
            r_norm = normalize_trajectory(traj)
            ax.plot(r_norm[:, 0], r_norm[:, 1], color=colors[category], alpha=0.5)
            ax.scatter(0, 0, color='black', s=20)  # Mark origin
    
    ax.set_title(f'Category {category} Normalized Trajectories')
    ax.set_xlabel('Normalized X')
    ax.set_ylabel('Normalized Y')
    ax.grid(True)
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('normalized_trajectories.png', dpi=300, bbox_inches='tight')

print("\nFeature exploration and visualization complete. Results saved as PNG files.")
