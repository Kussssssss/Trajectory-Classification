from pactus import Dataset, featurizers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from yupi.graphics import plot_2d, plot_hist
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.lines as mlines

# Load the dataset
SEED = 0
datasets = Dataset.hurdat2()
print(f"Loaded dataset: {datasets.name}")
print(f"Total trajectories: {len(datasets.trajs)}")
print(f"Different classes: {datasets.classes}")

# Analyze the first trajectory to understand structure
first_traj = datasets.trajs[0]
print("\nFirst trajectory analysis:")
print("Trajectory ID:", first_traj.traj_id)
print("Length (number of points):", len(first_traj))
print("Time (t) shape:", first_traj.t.shape)
print("Position (r) shape:", first_traj.r.shape)
print("Velocity (v) shape:", first_traj.v.shape)
print("Acceleration (a) shape:", first_traj.a.shape)
print("Delta t:", first_traj.dt)
print("Delta t Mean:", first_traj.dt_mean)
print("Delta t Standard:", first_traj.dt_std)
print("T_0:", first_traj.t_0)
print("Uniform Space:", first_traj.uniformly_spaced)

# Get trajectory bounds
min_lon, max_lon = first_traj.bounds[0]
min_lat, max_lat = first_traj.bounds[1]
print(f"\nLongitude range: from {min_lon} to {max_lon}")
print(f"Latitude range: from {min_lat} to {max_lat}")

# Examine the class information
print("\nExamining class information:")
print("Dataset classes:", datasets.classes)
print("First trajectory class label:", datasets.labels[0])

# Analyze class distribution
class_counts = {}
for i, traj in enumerate(datasets.trajs):
    class_label = datasets.labels[i]  # Get class from dataset labels
    if class_label in class_counts:
        class_counts[class_label] += 1
    else:
        class_counts[class_label] = 1

print("\nClass distribution:")
for class_label, count in sorted(class_counts.items()):
    print(f"Class {class_label}: {count} trajectories ({count/len(datasets.trajs)*100:.2f}%)")

# Analyze trajectory lengths
traj_lengths = [len(traj) for traj in datasets.trajs]
print("\nTrajectory length statistics:")
print(f"Min length: {min(traj_lengths)}")
print(f"Max length: {max(traj_lengths)}")
print(f"Average length: {sum(traj_lengths)/len(traj_lengths):.2f}")

# Analyze trajectory durations
traj_durations = [(traj.t[-1] - traj.t[0]) for traj in datasets.trajs]
print("\nTrajectory duration statistics (hours):")
print(f"Min duration: {min(traj_durations):.2f}")
print(f"Max duration: {max(traj_durations):.2f}")
print(f"Average duration: {sum(traj_durations)/len(traj_durations):.2f}")

# Analyze geographical distribution
all_lons = []
all_lats = []
for traj in datasets.trajs[:100]:  # Sample first 100 trajectories
    lons = traj.r[:, 0]
    lats = traj.r[:, 1]
    all_lons.extend(lons)
    all_lats.extend(lats)

print("\nGeographical distribution (sample of 100 trajectories):")
print(f"Longitude range: from {min(all_lons):.2f} to {max(all_lons):.2f}")
print(f"Latitude range: from {min(all_lats):.2f} to {max(all_lats):.2f}")

# Save a visualization of some sample trajectories
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

# Plot trajectories by class
class_colors = {
    0: 'blue',
    1: 'green',
    2: 'red',
    3: 'purple',
    4: 'orange',
    5: 'brown'
}

legend_handles = []
for class_label, color in class_colors.items():
    handle = mlines.Line2D([], [], color=color, marker='o', 
                          markersize=5, label=f'Class {class_label}')
    legend_handles.append(handle)

# Plot a sample of trajectories from each class
for class_label in datasets.classes:
    # Get indices of trajectories with this class
    class_indices = [i for i, label in enumerate(datasets.labels) if label == class_label]
    # Get sample of trajectories
    sample_size = min(5, len(class_indices))
    for idx in class_indices[:sample_size]:
        traj = datasets.trajs[idx]
        lons = traj.r[:, 0]
        lats = traj.r[:, 1]
        plt.plot(lons, lats, color=class_colors[class_label], alpha=0.7, 
                 linewidth=1.5, transform=ccrs.PlateCarree())
        plt.scatter(lons[0], lats[0], color=class_colors[class_label], 
                   marker='o', s=30, transform=ccrs.PlateCarree())

plt.legend(handles=legend_handles, title='Hurricane Categories')
plt.title('Sample Hurricane Trajectories by Category')
plt.grid(True)
plt.savefig('sample_trajectories.png', dpi=300, bbox_inches='tight')
print("\nSample trajectories visualization saved as 'sample_trajectories.png'")

# Create a dataset summary
print("\nCreating dataset summary...")
summary_data = {
    'Dataset Name': datasets.name,
    'Total Trajectories': len(datasets.trajs),
    'Classes': len(datasets.classes),
    'Min Trajectory Length': min(traj_lengths),
    'Max Trajectory Length': max(traj_lengths),
    'Avg Trajectory Length': sum(traj_lengths)/len(traj_lengths),
    'Min Duration (hours)': min(traj_durations),
    'Max Duration (hours)': max(traj_durations),
    'Avg Duration (hours)': sum(traj_durations)/len(traj_durations)
}

# Save summary to file
with open('dataset_summary.txt', 'w') as f:
    for key, value in summary_data.items():
        f.write(f"{key}: {value}\n")

print("Dataset analysis complete. Summary saved to 'dataset_summary.txt'")
