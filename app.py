import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
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

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = HurricaneDataProcessor()
    st.session_state.data_loaded = False
    st.session_state.features_extracted = False
    st.session_state.model_trained = False
    st.session_state.selected_trajectory = None

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
                'category': category
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
        legend_title_text='Hurricane Category'
    )
    
    return fig

# Function to create feature importance plot
def create_feature_importance_plot(model_results):
    feature_importance = model_results['feature_importance']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10), ax=ax)
    ax.set_title('Top 10 Feature Importance for Hurricane Category Prediction')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    
    return fig

# Function to create confusion matrix plot
def create_confusion_matrix_plot(model_results):
    cm = model_results['confusion_matrix']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=sorted(set(model_results['y_test'])),
                yticklabels=sorted(set(model_results['y_test'])))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    return fig

# Function to create feature distribution plot
def create_feature_distribution_plot(features_df, feature_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='category', y=feature_name, data=features_df, ax=ax,
                palette=['blue', 'green', 'red', 'purple', 'orange', 'brown'])
    ax.set_title(f'{feature_name} Distribution by Hurricane Category')
    ax.set_xlabel('Hurricane Category')
    ax.set_ylabel(feature_name)
    
    return fig

# Function to create normalized trajectory plot
def create_normalized_trajectory_plot(processor, category=None):
    # Get sample trajectories
    samples = processor.get_sample_trajectories(n_per_category=10)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, cat in enumerate(sorted(samples.keys())):
        ax = axes[i]
        
        # Skip if category filter is applied and doesn't match
        if category is not None and cat != category:
            continue
            
        for traj in samples[cat]:
            if len(traj) >= 3:  # Only plot trajectories with at least 3 points
                r_norm = processor.normalize_trajectory(traj)
                ax.plot(r_norm[:, 0], r_norm[:, 1], 
                        color=processor.get_category_color(cat), alpha=0.5)
                ax.scatter(0, 0, color='black', s=20)  # Mark origin
        
        ax.set_title(f'Category {cat} Normalized Trajectories')
        ax.set_xlabel('Normalized X')
        ax.set_ylabel('Normalized Y')
        ax.grid(True)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig

# Main app layout
def main():
    # Sidebar
    st.sidebar.title("Hurricane Analysis")
    
    # Data loading section in sidebar
    st.sidebar.header("Data")
    if st.sidebar.button("Load Hurricane Dataset"):
        with st.spinner("Loading hurricane dataset..."):
            dataset = load_data()
            st.sidebar.success(f"Loaded {len(dataset.trajs)} trajectories")
    
    # Only show these options if data is loaded
    if st.session_state.data_loaded:
        if st.sidebar.button("Extract Features"):
            with st.spinner("Extracting features..."):
                features_df = extract_features()
                st.sidebar.success(f"Extracted features from {len(features_df)} trajectories")
        
        if st.session_state.features_extracted and st.sidebar.button("Train Model"):
            with st.spinner("Training model..."):
                model_results = train_model()
                st.sidebar.success(f"Model trained with accuracy: {model_results['report']['accuracy']:.4f}")
    
    # Navigation in sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Trajectory Explorer", "Feature Analysis", "Prediction Model", "Trajectory Comparison"]
    )
    
    # Main content area
    if page == "Home":
        show_home_page()
    elif page == "Trajectory Explorer":
        show_trajectory_explorer()
    elif page == "Feature Analysis":
        show_feature_analysis()
    elif page == "Prediction Model":
        show_prediction_model()
    elif page == "Trajectory Comparison":
        show_trajectory_comparison()

# Home page
def show_home_page():
    st.title("Hurricane Trajectory Analysis and Prediction")
    st.write("""
    Welcome to the Hurricane Trajectory Analysis and Prediction application. This interactive dashboard allows you to explore
    hurricane trajectory data, visualize patterns, and predict hurricane categories based on trajectory features.
    """)
    
    # Dataset overview
    st.header("Dataset Overview")
    
    if not st.session_state.data_loaded:
        st.info("Please load the hurricane dataset using the button in the sidebar.")
    else:
        processor = st.session_state.processor
        summary = processor.get_dataset_summary()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Statistics")
            st.write(f"**Dataset Name:** {summary['dataset_name']}")
            st.write(f"**Total Trajectories:** {summary['total_trajectories']}")
            st.write(f"**Number of Classes:** {summary['classes']}")
            st.write(f"**Trajectory Length:** {summary['min_trajectory_length']} to {summary['max_trajectory_length']} points (avg: {summary['avg_trajectory_length']:.2f})")
            st.write(f"**Trajectory Duration:** {summary['min_duration_hours']:.2f} to {summary['max_duration_hours']:.2f} hours (avg: {summary['avg_duration_hours']:.2f})")
        
        with col2:
            st.subheader("Category Distribution")
            category_counts = summary['class_distribution']
            df_categories = pd.DataFrame({
                'Category': list(category_counts.keys()),
                'Count': list(category_counts.values())
            })
            df_categories['Percentage'] = df_categories['Count'] / df_categories['Count'].sum() * 100
            
            fig = px.bar(
                df_categories, 
                x='Category', 
                y='Count',
                color='Category',
                color_discrete_sequence=['blue', 'green', 'red', 'purple', 'orange', 'brown'],
                text='Percentage',
                labels={'Percentage': '%'},
                title='Hurricane Categories Distribution'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig)
    
    # Application sections
    st.header("Application Sections")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Trajectory Explorer")
        st.write("Visualize hurricane trajectories on an interactive map. Filter by category, and explore the geographical distribution of hurricanes.")
        
        st.subheader("Feature Analysis")
        st.write("Analyze the key features of hurricane trajectories, including velocity, trajectory length, and geographical extent. Explore correlations and distributions.")
    
    with col2:
        st.subheader("Prediction Model")
        st.write("Use machine learning to predict hurricane categories based on trajectory features. Explore model performance and feature importance.")
        
        st.subheader("Trajectory Comparison")
        st.write("Compare hurricane trajectories across different categories. Visualize normalized trajectory patterns and identify similarities.")

# Trajectory Explorer page
def show_trajectory_explorer():
    st.title("Hurricane Trajectory Explorer")
    
    if not st.session_state.data_loaded:
        st.info("Please load the hurricane dataset using the button in the sidebar.")
        return
    
    processor = st.session_state.processor
    
    # Filters
    st.sidebar.header("Filters")
    
    # Category filter
    categories = sorted(processor.dataset.classes)
    selected_categories = st.sidebar.multiselect(
        "Select Hurricane Categories",
        options=categories,
        default=categories
    )
    
    # Sample size slider
    sample_size = st.sidebar.slider(
        "Sample Size",
        min_value=10,
        max_value=200,
        value=50,
        step=10
    )
    
    # Filter trajectories by category
    filtered_indices = [i for i, label in enumerate(processor.dataset.labels) 
                       if label in selected_categories]
    filtered_trajs = [processor.dataset.trajs[i] for i in filtered_indices]
    filtered_labels = [processor.dataset.labels[i] for i in filtered_indices]
    
    st.write(f"Showing {min(sample_size, len(filtered_trajs))} trajectories out of {len(filtered_trajs)} filtered trajectories.")
    
    # Create and display map
    with st.spinner("Creating trajectory map..."):
        fig = create_trajectory_map(filtered_trajs, filtered_labels, sample_size)
        st.plotly_chart(fig, use_container_width=True)
    
    # Trajectory statistics
    st.header("Trajectory Statistics by Category")
    
    if st.session_state.features_extracted:
        features_df = processor.features_df
        
        # Filter features by selected categories
        filtered_features = features_df[features_df['category'].isin(selected_categories)]
        
        # Group by category
        grouped = filtered_features.groupby('category').agg({
            'traj_length': ['mean', 'min', 'max'],
            'traj_duration': ['mean', 'min', 'max'],
            'mean_velocity': ['mean', 'min', 'max'],
            'lon_range': ['mean', 'min', 'max'],
            'lat_range': ['mean', 'min', 'max']
        }).reset_index()
        
        # Flatten multi-index columns
        grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
        
        # Display statistics
        st.dataframe(grouped)
    else:
        st.info("Please extract features using the button in the sidebar to view trajectory statistics.")

# Feature Analysis page
def show_feature_analysis():
    st.title("Hurricane Feature Analysis")
    
    if not st.session_state.features_extracted:
        st.info("Please extract features using the button in the sidebar.")
        return
    
    processor = st.session_state.processor
    features_df = processor.features_df
    
    # Feature selection
    st.sidebar.header("Feature Selection")
    feature_options = [col for col in features_df.columns if col not in ['traj_id', 'category']]
    selected_feature = st.sidebar.selectbox(
        "Select Feature to Analyze",
        options=feature_options,
        index=feature_options.index('mean_velocity') if 'mean_velocity' in feature_options else 0
    )
    
    # Feature distribution
    st.header(f"{selected_feature} Distribution by Category")
    
    with st.spinner("Creating distribution plot..."):
        fig = create_feature_distribution_plot(features_df, selected_feature)
        st.pyplot(fig)
    
    # Feature correlation
    st.header("Feature Correlation Matrix")
    
    # Select features for correlation
    correlation_features = st.multiselect(
        "Select Features for Correlation Analysis",
        options=feature_options,
        default=feature_options[:5]
    )
    
    if correlation_features:
        # Add category for coloring
        corr_df = features_df[correlation_features + ['category']]
        
        # Calculate correlation
        corr_matrix = corr_df[correlation_features].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Feature Correlation Matrix')
        st.pyplot(fig)
    
    # Feature importance (if model is trained)
    if st.session_state.model_trained:
        st.header("Feature Importance for Hurricane Category Prediction")
        
        model_results = train_model()  # This will use cached results
        
        with st.spinner("Creating feature importance plot..."):
            fig = create_feature_importance_plot(model_results)
            st.pyplot(fig)

# Prediction Model page
def show_prediction_model():
    st.title("Hurricane Category Prediction Model")
    
    if not st.session_state.model_trained:
        st.info("Please train the model using the button in the sidebar.")
        return
    
    processor = st.session_state.processor
    model_results = train_model()  # This will use cached results
    
    # Model performance
    st.header("Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Classification Report")
        report = model_results['report']
        
        # Convert report to DataFrame for better display
        report_df = pd.DataFrame(report).transpose()
        
        # F
(Content truncated due to size limit. Use line ranges to read in chunks)