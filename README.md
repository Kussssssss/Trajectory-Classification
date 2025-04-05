# Hurricane Trajectory Analysis and Prediction

An interactive Streamlit application for analyzing hurricane trajectories, visualizing patterns, and predicting hurricane categories based on trajectory features.

## Overview

This application uses the Pactus library's hurricane dataset (HURDAT2) to provide insights into hurricane patterns and characteristics. It includes interactive visualizations, machine learning models for category prediction, and tools for comparing hurricane trajectories.

## Features

### Home Page
- Dataset overview with key statistics
- Hurricane category distribution
- Information about different hurricane categories

### Trajectory Explorer
- Interactive map visualization of hurricane trajectories
- 3D visualization of hurricane paths
- Animated hurricane trajectory visualization
- Velocity profile analysis
- Filtering by hurricane category

### Feature Analysis
- Visualization of key features by hurricane category
- Correlation analysis between features
- Feature distribution exploration
- Statistical insights on hurricane patterns

### Prediction Model
- Hurricane category prediction based on trajectory features
- Model performance metrics and visualization
- Feature importance analysis
- Interactive prediction for selected or custom trajectories

### Trajectory Comparison
- Normalized trajectory patterns by category
- Side-by-side comparison of multiple hurricanes
- Feature comparison between hurricanes
- Similarity analysis

### Hurricane Impact
- Visualization of potential impact areas based on trajectory and category
- Risk assessment information
- Historical comparison with similar hurricanes

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd hurricane-app
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Run the application:
```
streamlit run app.py
```

For enhanced features with additional visualizations:
```
streamlit run app_enhanced.py
```

## Usage Guide

### Getting Started

1. Load the hurricane dataset using the "Load Hurricane Dataset" button in the sidebar.
2. Extract features using the "Extract Features" button.
3. Train the prediction model using the "Train Model" button.
4. Navigate between different sections using the sidebar.

### Data Processing

The application uses the `HurricaneDataProcessor` class to handle data loading, feature extraction, and model training. Key features extracted from hurricane trajectories include:

- Mean longitude and latitude
- Velocity and acceleration statistics
- Trajectory length and duration
- Geographical extent

### Machine Learning Model

The application uses a Random Forest classifier to predict hurricane categories based on trajectory features. The model is trained on extracted features and evaluated using standard metrics like accuracy, precision, recall, and F1-score.

## Technical Details

### Dependencies

- Pactus: For hurricane trajectory dataset and analysis
- Streamlit: For interactive web application
- Plotly: For interactive visualizations
- Scikit-learn: For machine learning models
- Pandas & NumPy: For data processing
- Matplotlib & Seaborn: For static visualizations
- Cartopy: For geographical visualizations

### Dataset

The HURDAT2 dataset contains 1,903 hurricane trajectories across 6 categories (0-5):
- Category 0: Tropical Depression (49.97%)
- Category 1: Tropical Storm (19.81%)
- Category 2: Category 1 Hurricane (13.03%)
- Category 3: Category 2 Hurricane (8.51%)
- Category 4: Category 3 Hurricane (6.67%)
- Category 5: Category 4-5 Hurricane (2.00%)

## Accessing the Application

The application is currently deployed and accessible at:
https://8501-i5ai3pk5d8p96uexogyft-86c9dbe1.manus.computer

## Future Enhancements

Potential future enhancements for this application include:
- Integration with real-time hurricane data
- More advanced machine learning models
- Climate change analysis features
- Mobile-optimized interface
- User account system for saving analyses

## License

This project is licensed under the MIT License - see the LICENSE file for details.
