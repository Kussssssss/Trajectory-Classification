import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from data_processing import HurricaneDataProcessor
import os
import pickle

# Set page configuration
st.set_page_config(
    page_title="Hurricane Trajectory Analysis",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #616161;
    }
    .category-0 { color: blue; }
    .category-1 { color: green; }
    .category-2 { color: red; }
    .category-3 { color: purple; }
    .category-4 { color: orange; }
    .category-5 { color: brown; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = HurricaneDataProcessor()
    st.session_state.data_loaded = False
    st.session_state.features_extracted = False
    st.session_state.model_trained = False
    st.session_state.selected_trajectory = None
    st.session_state.animation_speed = 50
    st.session_state.animation_frame = 0
    st.session_state.show_animation = False

# Function to load data
@st.cache_resource
def load_data():
    processor = st.session_state.processor
    dataset = processor.load_data()
    st.session_state.data_loaded = True
    return dataset

# Function to extract features
@st.cache_data
def extract_features():
    processor = st.session_state.processor
    features_df = processor.extract_features()
    st.session_state.features_extracted = True
    return features_df

# Function to train model
@st.cache_resource
def train_model():
    processor = st.session_state.processor
    model_results = processor.train_model()
    st.session_state.model_trained = True
    # Save model for future use
    processor.save_model()
    return model_results

# Function to create trajectory map
def create_trajectory_map(trajectories, labels, sample_size=50):
    # Sample trajectories if there are too many
    if len(trajectories) > sample_size:
        indices = np.random.choice(len(trajectories), sample_size, replace=False)
        sample_trajs = [trajectories[i] for i in indices]
        sample_labels = [labels[i] for i in indices]
    else:
        sample_trajs = trajectories
        sample_labels = labels
    
    # Create dataframe for plotting
    df_points = []
    
    for i, traj in enumerate(sample_trajs):
        category = sample_labels[i]
        for j in range(len(traj.r)):
            df_points.append({
                'traj_id': traj.traj_id,
                'point_id': j,
                'longitude': traj.r[j, 0],
                'latitude': traj.r[j, 1],
                'category': category,
                'time_step': j
            })
    
    df = pd.DataFrame(df_points)
    
    # Create map
    fig = px.line_geo(
        df, 
        lat='latitude', 
        lon='longitude',
        color='category',
        color_discrete_sequence=['blue', 'green', 'red', 'purple', 'orange', 'brown'],
        line_group='traj_id',
        title='Hurricane Trajectories by Category'
    )
    
    # Add markers for starting points
    start_points = df[df['point_id'] == 0]
    fig.add_trace(
        go.Scattergeo(
            lat=start_points['latitude'],
            lon=start_points['longitude'],
            mode='markers',
            marker=dict(size=6, color=start_points['category'], 
                        colorscale=['blue', 'green', 'red', 'purple', 'orange', 'brown']),
            name='Starting Points'
        )
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        legend_title_text='Hurricane Category',
        geo=dict(
            showland=True,
            landcolor='rgb(217, 217, 217)',
            coastlinecolor='rgb(37, 102, 142)',
            countrycolor='rgb(217, 217, 217)',
            showocean=True,
            oceancolor='rgb(204, 229, 255)',
            showlakes=True,
            lakecolor='rgb(204, 229, 255)',
            showrivers=True,
            rivercolor='rgb(204, 229, 255)'
        )
    )
    
    return fig, df

# Function to create animated trajectory map
def create_animated_trajectory_map(df):
    # Create animated map
    fig = px.line_geo(
        df, 
        lat='latitude', 
        lon='longitude',
        color='category',
        color_discrete_sequence=['blue', 'green', 'red', 'purple', 'orange', 'brown'],
        line_group='traj_id',
        animation_frame='time_step',
        title='Hurricane Trajectories Animation'
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        legend_title_text='Hurricane Category',
        geo=dict(
            showland=True,
            landcolor='rgb(217, 217, 217)',
            coastlinecolor='rgb(37, 102, 142)',
            countrycolor='rgb(217, 217, 217)',
            showocean=True,
            oceancolor='rgb(204, 229, 255)',
            showlakes=True,
            lakecolor='rgb(204, 229, 255)',
            showrivers=True,
            rivercolor='rgb(204, 229, 255)'
        ),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True}]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}]
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 10},
            'x': 0.1,
            'y': 0
        }]
    )
    
    return fig

# Function to create 3D trajectory visualization
def create_3d_trajectory_plot(trajectories, labels, sample_size=20):
    # Sample trajectories if there are too many
    if len(trajectories) > sample_size:
        indices = np.random.choice(len(trajectories), sample_size, replace=False)
        sample_trajs = [trajectories[i] for i in indices]
        sample_labels = [labels[i] for i in indices]
    else:
        sample_trajs = trajectories
        sample_labels = labels
    
    # Create dataframe for plotting
    df_points = []
    
    for i, traj in enumerate(sample_trajs):
        category = sample_labels[i]
        for j in range(len(traj.r)):
            # Calculate time as percentage of trajectory duration
            time_pct = j / (len(traj.r) - 1) if len(traj.r) > 1 else 0
            
            df_points.append({
                'traj_id': traj.traj_id,
                'point_id': j,
                'longitude': traj.r[j, 0],
                'latitude': traj.r[j, 1],
                'time': time_pct,  # Use time as z-axis
                'category': category
            })
    
    df = pd.DataFrame(df_points)
    
    # Create 3D plot
    fig = px.line_3d(
        df, 
        x='longitude', 
        y='latitude',
        z='time',
        color='category',
        color_discrete_sequence=['blue', 'green', 'red', 'purple', 'orange', 'brown'],
        line_group='traj_id',
        title='3D Hurricane Trajectories (Z-axis represents time)'
    )
    
    # Add markers for starting points
    start_points = df[df['point_id'] == 0]
    fig.add_trace(
        go.Scatter3d(
            x=start_points['longitude'],
            y=start_points['latitude'],
            z=start_points['time'],
            mode='markers',
            marker=dict(size=4, color=start_points['category'], 
                        colorscale=['blue', 'green', 'red', 'purple', 'orange', 'brown']),
            name='Starting Points'
        )
    )
    
    # Update layout
    fig.update_layout(
        height=700,
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Time (normalized)',
            aspectmode='manual',
            aspectratio=dict(x=1.5, y=1, z=0.5)
        )
    )
    
    return fig

# Function to create feature importance plot
def create_feature_importance_plot(model_results):
    feature_importance = model_results['feature_importance']
    
    fig = px.bar(
        feature_importance.head(10),
        x='importance',
        y='feature',
        orientation='h',
        color='importance',
        color_continuous_scale='Blues',
        title='Top 10 Feature Importance for Hurricane Category Prediction'
    )
    
    fig.update_layout(
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=500
    )
    
    return fig

# Function to create confusion matrix plot
def create_confusion_matrix_plot(model_results):
    cm = model_results['confusion_matrix']
    
    # Create labels for categories
    categories = sorted(set(model_results['y_test']))
    labels = [f'Category {cat}' for cat in categories]
    
    # Create heatmap
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        color_continuous_scale='Blues',
        labels=dict(x='Predicted', y='True', color='Count'),
        title='Confusion Matrix'
    )
    
    # Add text annotations
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            fig.add_annotation(
                x=labels[j],
                y=labels[i],
                text=str(cm[i, j]),
                showarrow=False,
                font=dict(color='white' if cm[i, j] > cm.max()/2 else 'black')
            )
    
    fig.update_layout(height=500)
    
    return fig

# Function to create feature distribution plot
def create_feature_distribution_plot(features_df, feature_name):
    fig = px.box(
        features_df,
        x='category',
        y=feature_name,
        color='category',
        color_discrete_sequence=['blue', 'green', 'red', 'purple', 'orange', 'brown'],
        title=f'{feature_name} Distribution by Hurricane Category',
        labels={'category': 'Hurricane Category', feature_name: feature_name.replace('_', ' ').title()}
    )
    
    fig.update_layout(height=500)
    
    return fig

# Function to create normalized trajectory plot
def create_normalized_trajectory_plot(processor, category=None):
    # Get sample trajectories
    samples = processor.get_sample_trajectories(n_per_category=10)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f'Category {cat}' for cat in sorted(samples.keys())],
        specs=[[{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}],
               [{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}]]
    )
    
    # Add traces for each category
    for i, cat in enumerate(sorted(samples.keys())):
        row = i // 3 + 1
        col = i % 3 + 1
        
        # Skip if category filter is applied and doesn't match
        if category is not None and cat != category:
            continue
            
        for traj in samples[cat]:
            if len(traj) >= 3:  # Only plot trajectories with at least 3 points
                r_norm = processor.normalize_trajectory(traj)
                
                # Add trajectory line
                fig.add_trace(
                    go.Scatter(
                        x=r_norm[:, 0],
                        y=r_norm[:, 1],
                        mode='lines',
                        line=dict(color=processor.get_category_color(cat), width=1.5),
                        opacity=0.7,
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                # Add origin point
                fig.add_trace(
                    go.Scatter(
                        x=[0],
                        y=[0],
                        mode='markers',
                        marker=dict(color='black', size=6),
                        showlegend=False
                    ),
                    row=row, col=col
                )
    
    # Update layout
    fig.update_layout(
        height=700,
        title='Normalized Hurricane Trajectories by Category',
        showlegend=False
    )
    
    # Update axes properties
    for i in range(1, 7):
        row = (i-1) // 3 + 1
        col = (i-1) % 3 + 1
        
        fig.update_xaxes(title_text='Normalized X', row=row, col=col, zeroline=True, zerolinewidth=1, zerolinecolor='gray')
        fig.update_yaxes(title_text='Normalized Y', row=row, col=col, zeroline=True, zerolinewidth=1, zerolinecolor='gray')
    
    return fig

# Function to create velocity profile visualization
def create_velocity_profile(trajectories, labels, sample_size=10):
    # Sample trajectories if there are too many
    if len(trajectories) > sample_size:
        indices = np.random.choice(len(trajectories), sample_size, replace=False)
        sample_trajs = [trajectories[i] for i in indices]
        sample_labels = [labels[i] for i in indices]
    else:
        sample_trajs = trajectories
        sample_labels = labels
    
    # Create figure
    fig = make_subplots(rows=len(sample_trajs), cols=1, 
                        shared_xaxes=True,
                        subplot_titles=[f'Trajectory {traj.traj_id} (Category {label})' 
                                        for traj, label in zip(sample_trajs, sample_labels)])
    
    # Colors for categories
    category_colors = {
        0: 'blue',
        1: 'green',
        2: 'red',
        3: 'purple',
        4: 'orange',
        5: 'brown'
    }
    
    # Add velocity profiles
    for i, (traj, label) in enumerate(zip(sample_trajs, sample_labels)):
        try:
            # Calculate velocity magnitude
            v_magnitude = np.sqrt(np.sum(traj.v**2, axis=1))
            
            # Normalize time to percentage
            time_pct = np.linspace(0, 100, len(v_magnitude))
            
            # Add trace
            fig.add_trace(
                go.Scatter(
                    x=time_pct,
                    y=v_magnitude,
                    mode='lines',
                    line=dict(color=category_colors.get(label, 'gray'), width=2),
                    name=f'Category {label}'
                ),
                row=i+1, col=1
            )
            
            # Add mean velocity line
            mean_v = np.mean(v_magnitude)
            fig.add_trace(
                go.Scatter(
                    x=[0, 100],
                    y=[mean_v, mean_v],
                    mode='lines',
                    line=dict(color='black', width=1, dash='dash'),
                    name='Mean Velocity',
                    showlegend=False
                ),
                row=i+1, col=1
            )
            
        except ValueError:
            # Skip trajectories too short to estimate velocity
            continue
    
    # Update layout
    fig.update_layout(
        height=100 * len(sample_trajs),
        title='Hurricane Velocity Profiles',
        showlegend=True
    )
    
    # Update axes
    for i in range(len(sample_trajs)):
        fig.update_yaxes(title_text='Velocity', row=i+1, col=1)
    
    fig.update_xaxes(title_text='Trajectory Progress (%)', row=len(sample_trajs), col=1)
    
    return fig

# Function to create hurricane impact visualization
def create_impac
(Content truncated due to size limit. Use line ranges to read in chunks)