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
import folium
from streamlit_folium import st_folium

# Cấu hình trang
st.set_page_config(
    page_title="Phân tích quỹ đạo bão và dự đoán",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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

# Khởi tạo session state
if 'processor' not in st.session_state:
    st.session_state.processor = HurricaneDataProcessor()
    st.session_state.data_loaded = False
    st.session_state.features_extracted = False
    st.session_state.model_trained = False
    st.session_state.selected_trajectory = None
    st.session_state.animation_speed = 50
    st.session_state.animation_frame = 0
    st.session_state.show_animation = False
    st.session_state.preprocessing_options = {
        'outlier_method': 'winsorize',
        'create_interactions': True,
        'use_features': True
    }

# --- Helper: Chuyển DataFrame sang định dạng Arrow-compatible ---
def make_dataframe_arrow_compatible(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            non_null = df[col].dropna()
            if not non_null.empty:
                sample = non_null.iloc[0]
                if isinstance(sample, (list, np.ndarray)):
                    df[col] = df[col].apply(lambda x: np.mean(x) if isinstance(x, (list, np.ndarray)) else x)
    return df

# --- Các hàm xử lý dữ liệu ---
@st.cache_resource
def load_data():
    processor = st.session_state.processor
    dataset = processor.load_data()
    st.session_state.data_loaded = True
    return dataset

@st.cache_data
def extract_features(_use_features=True, _outlier_method='winsorize', _create_interactions=True):
    processor = st.session_state.processor
    
    if _use_features:
        features_df = processor.process_data_pipeline(
            outlier_method=_outlier_method,
            create_interactions=_create_interactions
        )
    else:
        features_df = processor.extract_features()
    
    st.session_state.features_extracted = True
    
    return make_dataframe_arrow_compatible(features_df)

@st.cache_resource
def train_model(_use_features=True):
    processor = st.session_state.processor    
    model_results = processor.train_model(
        use_features=_use_features
    )
    
    st.session_state.model_trained = True
    processor.save_model()
    return model_results

# --- Các hàm trực quan hóa ---
def create_trajectory_map(trajectories, labels, sample_size=50):
    if len(trajectories) > sample_size:
        indices = np.random.choice(len(trajectories), sample_size, replace=False)
        sample_trajs = [trajectories[i] for i in indices]
        sample_labels = [labels[i] for i in indices]
    else:
        sample_trajs = trajectories
        sample_labels = labels

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
    fig = px.line_geo(
        df, 
        lat='latitude', 
        lon='longitude',
        color='category',
        color_discrete_sequence=['blue', 'green', 'red', 'purple', 'orange', 'brown'],
        line_group='traj_id',
        title='Quỹ đạo bão theo loại'
    )
    start_points = df[df['point_id'] == 0]
    fig.add_trace(
        go.Scattergeo(
            lat=start_points['latitude'],
            lon=start_points['longitude'],
            mode='markers',
            marker=dict(size=6, color=start_points['category'],
                        colorscale=['blue', 'green', 'red', 'purple', 'orange', 'brown']),
            name='Điểm khởi đầu'
        )
    )
    fig.update_layout(
        height=600,
        legend_title_text='Loại bão',
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

def create_animated_trajectory_map(df):
    # Tích lũy các điểm: với mỗi frame f, hiển thị các điểm có time_step <= f
    max_frame = int(df['time_step'].max())
    df_list = []
    for f in range(max_frame + 1):
        temp = df[df['time_step'] <= f].copy()
        temp['frame'] = f
        df_list.append(temp)
    df_accumulated = pd.concat(df_list)
    
    fig = px.line_geo(
        df_accumulated, 
        lat='latitude', 
        lon='longitude',
        color='category',
        color_discrete_sequence=['blue', 'green', 'red', 'purple', 'orange', 'brown'],
        line_group='traj_id',
        animation_frame='frame',
        title='Animation quỹ đạo bão'
    )
    fig.update_layout(
        height=600,
        legend_title_text='Loại bão',
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

def create_3d_trajectory_plot(trajectories, labels, sample_size=20):
    if len(trajectories) > sample_size:
        indices = np.random.choice(len(trajectories), sample_size, replace=False)
        sample_trajs = [trajectories[i] for i in indices]
        sample_labels = [labels[i] for i in indices]
    else:
        sample_trajs = trajectories
        sample_labels = labels

    df_points = []
    for i, traj in enumerate(sample_trajs):
        category = sample_labels[i]
        for j in range(len(traj.r)):
            time_pct = j / (len(traj.r)-1) if len(traj.r) > 1 else 0
            df_points.append({
                'traj_id': traj.traj_id,
                'point_id': j,
                'longitude': traj.r[j, 0],
                'latitude': traj.r[j, 1],
                'time': time_pct,
                'category': category
            })
    df = pd.DataFrame(df_points)
    fig = px.line_3d(
        df,
        x='longitude',
        y='latitude',
        z='time',
        color='category',
        color_discrete_sequence=['blue', 'green', 'red', 'purple', 'orange', 'brown'],
        line_group='traj_id',
        title='3D Quỹ đạo bão (trục Z: thời gian chuẩn hóa)'
    )
    start_points = df[df['point_id'] == 0]
    fig.add_trace(
        go.Scatter3d(
            x=start_points['longitude'],
            y=start_points['latitude'],
            z=start_points['time'],
            mode='markers',
            marker=dict(size=4, color=start_points['category'],
                        colorscale=['blue', 'green', 'red', 'purple', 'orange', 'brown']),
            name='Điểm khởi đầu'
        )
    )
    fig.update_layout(
        height=700,
        scene=dict(
            xaxis_title='Kinh độ',
            yaxis_title='Vĩ độ',
            zaxis_title='Thời gian (chuẩn hóa)',
            aspectmode='manual',
            aspectratio=dict(x=1.5, y=1, z=0.5)
        )
    )
    return fig

def create_velocity_profile(trajectories, labels, sample_size=10):
    if len(trajectories) > sample_size:
        indices = np.random.choice(len(trajectories), sample_size, replace=False)
        sample_trajs = [trajectories[i] for i in indices]
        sample_labels = [labels[i] for i in indices]
    else:
        sample_trajs = trajectories
        sample_labels = labels

    fig = make_subplots(rows=len(sample_trajs), cols=1, 
                        shared_xaxes=True,
                        subplot_titles=[f'Trajectory {traj.traj_id} (Loại {label})' 
                                        for traj, label in zip(sample_trajs, sample_labels)])
    category_colors = {0: 'blue', 1: 'green', 2: 'red', 3: 'purple', 4: 'orange', 5: 'brown'}
    for i, (traj, label) in enumerate(zip(sample_trajs, sample_labels)):
        try:
            v_magnitude = np.sqrt(np.sum(traj.v**2, axis=1))
            time_pct = np.linspace(0, 100, len(v_magnitude))
            fig.add_trace(
                go.Scatter(
                    x=time_pct,
                    y=v_magnitude,
                    mode='lines',
                    line=dict(color=category_colors.get(label, 'gray'), width=2),
                    name=f'Loại {label}'
                ),
                row=i+1, col=1
            )
            mean_v = np.mean(v_magnitude)
            fig.add_trace(
                go.Scatter(
                    x=[0, 100],
                    y=[mean_v, mean_v],
                    mode='lines',
                    line=dict(color='black', width=1, dash='dash'),
                    name='Vận tốc trung bình',
                    showlegend=False
                ),
                row=i+1, col=1
            )
        except ValueError:
            continue
    fig.update_layout(
        height=100 * len(sample_trajs),
        title='Biểu đồ vận tốc của quỹ đạo bão',
        showlegend=True
    )
    for i in range(len(sample_trajs)):
        fig.update_yaxes(title_text='Vận tốc', row=i+1, col=1)
    fig.update_xaxes(title_text='Tiến trình trajectory (%)', row=len(sample_trajs), col=1)
    return fig
# --- Hàm tính khoảng cách theo Haversine ---
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # bán kính Trái đất km
    return c * r

# --- Trang dự đoán qua vẽ đường đi của bão ---
def show_drawing_prediction():
    st.title("Dự đoán loại bão từ đường vẽ")
    st.write("Trên bản đồ dưới đây, hãy sử dụng công cụ vẽ (Polyline) để vẽ đường đi của bão quanh khu vực châu Mỹ.")
    
    # Tạo bản đồ với folium, trung tâm châu Mỹ
    m = folium.Map(location=[37, -95], zoom_start=4)
    draw = folium.plugins.Draw(
        export=True,
        draw_options={
            'polyline': True,
            'polygon': False,
            'circle': False,
            'rectangle': False,
            'marker': False,
            'circlemarker': False
        }
    )
    draw.add_to(m)
    output = st_folium(m, width=700, height=500)
    
    if output and output.get('last_active_drawing'):
        geojson = output['last_active_drawing']
        if geojson['geometry']['type'] == 'LineString':
            coords = geojson['geometry']['coordinates']  # [ [lon, lat], ... ]
            st.subheader("Đường bạn vẽ:")
            st.write(coords)
            
            # Tính các đặc trưng từ đường vẽ:
            # Giả sử thời gian giữa các điểm là 1 giờ
            total_distance = 0
            speeds = []
            lons = []
            lats = []
            for i in range(len(coords)-1):
                lon1, lat1 = coords[i]
                lon2, lat2 = coords[i+1]
                d = haversine(lon1, lat1, lon2, lat2)
                total_distance += d
                speeds.append(d)  # km/h
                lons.extend([lon1, lon2])
                lats.extend([lat1, lat2])
            average_speed = np.mean(speeds) if speeds else 0
            max_speed = np.max(speeds) if speeds else 0
            lon_range = max(lons) - min(lons) if lons else 0
            lat_range = max(lats) - min(lats) if lats else 0
            
            st.write(f"**Tổng quãng đường:** {total_distance:.2f} km")
            st.write(f"**Vận tốc trung bình:** {average_speed:.2f} km/h")
            st.write(f"**Vận tốc tối đa:** {max_speed:.2f} km/h")
            st.write(f"**Khoảng cách kinh độ:** {lon_range:.2f} độ")
            st.write(f"**Khoảng cách vĩ độ:** {lat_range:.2f} độ")
            
            # Tạo vector đặc trưng: giả sử mô hình yêu cầu các đặc trưng này
            feature_vector = np.array([total_distance, average_speed, max_speed, lon_range, lat_range]).reshape(1, -1)
            st.write("Vector đặc trưng:")
            st.write(feature_vector)
            
            # Dự đoán loại bão (giả sử mô hình đã được huấn luyện và lưu trong processor.model)
            try:
                prediction = st.session_state.processor.model.predict(feature_vector)[0]
                st.success(f"Dự đoán loại bão: {prediction}")
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {e}")
            
            # Hiển thị đường vẽ lên bản đồ (sử dụng folium)
            m2 = folium.Map(location=[np.mean(lats), np.mean(lons)], zoom_start=5)
            folium.PolyLine(locations=[(lat, lon) for lon, lat in coords], color='red', weight=3).add_to(m2)
            st.subheader("Đường vẽ trên bản đồ:")
            st_folium(m2, width=700, height=500)
        else:
            st.error("Vui lòng vẽ một đường polyline.")
    else:
        st.info("Vui lòng vẽ đường đi của bão trên bản đồ.")

def create_feature_importance_plot(model_results):
    feature_importance = model_results['feature_importance']
    fig = px.bar(
        feature_importance.head(10),
        x='importance',
        y='feature',
        orientation='h',
        color='importance',
        color_continuous_scale='Blues',
        title='Top 10 Tầm quan trọng của đặc trưng cho dự đoán loại bão'
    )
    fig.update_layout(
        xaxis_title='Tầm quan trọng',
        yaxis_title='Đặc trưng',
        height=500
    )
    return fig

def create_confusion_matrix_plot(model_results):
    cm = model_results['confusion_matrix']
    categories = sorted(set(model_results['y_test']))
    labels = [f'Loại {cat}' for cat in categories]
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        color_continuous_scale='Blues',
        labels=dict(x='Dự đoán', y='Thật', color='Số lượng'),
        title='Ma trận nhầm lẫn'
    )
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

def create_feature_distribution_plot(features_df, feature_name):
    fig = px.box(
        features_df,
        x='category',
        y=feature_name,
        color='category',
        color_discrete_sequence=['blue', 'green', 'red', 'purple', 'orange', 'brown'],
        title=f'Phân phối {feature_name} theo loại bão',
        labels={'category': 'Loại bão', feature_name: feature_name.replace('_', ' ').title()}
    )
    fig.update_layout(height=500)
    return fig

def create_normalized_trajectory_plot(processor, category=None):
    samples = processor.get_sample_trajectories(n_per_category=10)
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f'Loại {cat}' for cat in sorted(samples.keys())],
        specs=[[{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}],
               [{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}]]
    )
    for i, cat in enumerate(sorted(samples.keys())):
        row = i // 3 + 1
        col = i % 3 + 1
        if category is not None and cat != category:
            continue
        for traj in samples[cat]:
            if len(traj) >= 3:
                r_norm = processor.normalize_trajectory(traj)
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
        fig.update_xaxes(title_text='X chuẩn hóa', row=row, col=col, zeroline=True, zerolinewidth=1, zerolinecolor='gray')
        fig.update_yaxes(title_text='Y chuẩn hóa', row=row, col=col, zeroline=True, zerolinewidth=1, zerolinecolor='gray')
    fig.update_layout(
        height=700,
        title='Quỹ đạo chuẩn hóa theo loại bão',
        showlegend=False
    )
    return fig

def create_hurricane_impact_visualization(features_df):
    if 'impact_score' in features_df.columns:
        fig = px.histogram(features_df, x='impact_score', color='category', 
                           color_discrete_sequence=['blue', 'green', 'red', 'purple', 'orange', 'brown'],
                           title='Phân bố Impact Score theo loại bão')
    else:
        fig = px.histogram(features_df, x='traj_duration', color='category', 
                           color_discrete_sequence=['blue', 'green', 'red', 'purple', 'orange', 'brown'],
                           title='Phân bố Thời lượng trajectory theo loại bão')
    fig.update_layout(height=500)
    return fig

# --- Các hàm giao diện dự đoán dữ liệu đầu vào thực tế ---
def show_real_input_prediction():
    st.title("Dự đoán từ dữ liệu đầu vào thực tế")
    st.write("Tải lên file CSV chứa dữ liệu trajectory với các cột: t, longitude, latitude")
    uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.subheader("Dữ liệu đầu vào:")
            st.dataframe(df_input)
            # Tạo đối tượng trajectory từ dữ liệu CSV
            # Giả sử file CSV có các cột: t, longitude, latitude
            traj_input = type("Trajectory", (object,), {})()
            traj_input.t = df_input["t"].values
            traj_input.r = df_input[["longitude", "latitude"]].values
            # Chuyển đổi dữ liệu trajectory thành vector đặc trưng sử dụng featurizer của mô hình
            features = st.session_state.processor.model.featurizer.transform(traj_input)
            prediction = st.session_state.processor.model.predict([features])[0]
            st.success(f"Dự đoán loại bão: {prediction}")
            # Hiển thị trajectory lên bản đồ
            fig, _ = create_trajectory_map([traj_input], [prediction], sample_size=1)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Lỗi xử lý dữ liệu đầu vào: {e}")

# --- Các trang giao diện chính ---
def show_home_page():
    st.title("Phân tích quỹ đạo bão và dự đoán")
    st.write("""
    Chào mừng bạn đến với ứng dụng phân tích quỹ đạo bão và dự đoán loại bão. Dashboard này cho phép bạn khám phá dữ liệu quỹ đạo bão,
    trực quan hóa các mẫu và dự đoán loại bão dựa trên đặc trưng quỹ đạo.
    """)
    st.header("Tổng quan dữ liệu")
    if not st.session_state.data_loaded:
        st.info("Vui lòng load dữ liệu bão bằng nút ở thanh bên.")
    else:
        processor = st.session_state.processor
        summary = processor.get_dataset_summary()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Thống kê dữ liệu")
            st.write(f"**Tên dataset:** {summary['dataset_name']}")
            st.write(f"**Tổng số trajectory:** {summary['total_trajectories']}")
            st.write(f"**Số loại bão:** {summary['classes']}")
            st.write(f"**Độ dài trajectory:** từ {summary['min_trajectory_length']} đến {summary['max_trajectory_length']} điểm (trung bình: {summary['avg_trajectory_length']:.2f})")
            st.write(f"**Thời lượng trajectory:** từ {summary['min_duration_hours']:.2f} đến {summary['max_duration_hours']:.2f} giờ (trung bình: {summary['avg_duration_hours']:.2f})")
        with col2:
            st.subheader("Phân bố loại bão")
            category_counts = summary['class_distribution']
            df_categories = pd.DataFrame({
                'Loại': list(category_counts.keys()),
                'Số lượng': list(category_counts.values())
            })
            df_categories['Phần trăm'] = df_categories['Số lượng'] / df_categories['Số lượng'].sum() * 100
            fig = px.bar(
                df_categories,
                x='Loại',
                y='Số lượng',
                color='Loại',
                color_discrete_sequence=['blue', 'green', 'red', 'purple', 'orange', 'brown'],
                text='Phần trăm',
                labels={'Phần trăm': '%'},
                title='Phân bố loại bão'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
    st.header("Các mục trong ứng dụng")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Trình duyệt quỹ đạo")
        st.write("Trực quan hóa quỹ đạo bão trên bản đồ tương tác và khám phá phân bố địa lý theo loại.")
        st.subheader("Phân tích đặc trưng")
        st.write("Phân tích các đặc trưng (vận tốc, độ dài, thời lượng, v.v.) và khám phá mối tương quan.")
    with col2:
        st.subheader("Mô hình dự đoán")
        st.write("Dự đoán loại bão dựa trên đặc trưng quỹ đạo và đánh giá hiệu năng mô hình.")
        st.subheader("So sánh quỹ đạo")
        st.write("So sánh quỹ đạo chuẩn hóa giữa các loại bão.")

def show_trajectory_explorer():
    st.title("Trình duyệt quỹ đạo bão")
    if not st.session_state.data_loaded:
        st.info("Vui lòng load dữ liệu bão bằng nút ở thanh bên.")
        return
    processor = st.session_state.processor
    st.sidebar.header("Bộ lọc")
    categories = sorted(processor.dataset.classes)
    selected_categories = st.sidebar.multiselect(
        "Chọn loại bão",
        options=categories,
        default=categories
    )
    sample_size = st.sidebar.slider(
        "Kích thước mẫu",
        min_value=10,
        max_value=200,
        value=50,
        step=10
    )
    filtered_indices = [i for i, label in enumerate(processor.dataset.labels) if label in selected_categories]
    filtered_trajs = [processor.dataset.trajs[i] for i in filtered_indices]
    filtered_labels = [processor.dataset.labels[i] for i in filtered_indices]
    st.write(f"Hiển thị {min(sample_size, len(filtered_trajs))} trajectory trên tổng số {len(filtered_trajs)} trajectory đã lọc.")
    with st.spinner("Tạo bản đồ quỹ đạo..."):
        fig, _ = create_trajectory_map(filtered_trajs, filtered_labels, sample_size)
        st.plotly_chart(fig, use_container_width=True)
    st.header("Thống kê trajectory theo loại")
    if st.session_state.features_extracted:
        features_df = st.session_state.processor.features_df.copy()
        features_df = make_dataframe_arrow_compatible(features_df)
        filtered_features = features_df[features_df['category'].isin(selected_categories)]
        grouped = filtered_features.groupby('category').agg({
            'traj_length': ['mean', 'min', 'max'],
            'traj_duration': ['mean', 'min', 'max'],
            'mean_velocity': ['mean', 'min', 'max'],
            'lon_range': ['mean', 'min', 'max'],
            'lat_range': ['mean', 'min', 'max']
        }).reset_index()
        grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
        grouped = make_dataframe_arrow_compatible(grouped)
        st.dataframe(grouped)
    else:
        st.info("Vui lòng trích xuất đặc trưng để xem thống kê trajectory.")

def show_feature_analysis():
    st.title("Phân tích đặc trưng bão")
    if not st.session_state.features_extracted:
        st.info("Vui lòng trích xuất đặc trưng bằng nút ở thanh bên.")
        return
    processor = st.session_state.processor
    features_df = st.session_state.processor.features_df.copy()
    features_df = make_dataframe_arrow_compatible(features_df)
    st.sidebar.header("Chọn đặc trưng")
    feature_options = [col for col in features_df.columns if col not in ['traj_id', 'category']]
    selected_feature = st.sidebar.selectbox(
        "Chọn đặc trưng cần phân tích",
        options=feature_options,
        index=feature_options.index('mean_velocity') if 'mean_velocity' in feature_options else 0
    )
    st.header(f"Phân phối {selected_feature} theo loại bão")
    with st.spinner("Tạo biểu đồ phân phối..."):
        fig = create_feature_distribution_plot(features_df, selected_feature)
        st.plotly_chart(fig, use_container_width=True)
    st.header("Ma trận tương quan của đặc trưng")
    correlation_features = st.multiselect(
        "Chọn các đặc trưng để phân tích tương quan",
        options=feature_options,
        default=feature_options[:5]
    )
    if correlation_features:
        corr_df = features_df[correlation_features + ['category']]
        corr_matrix = corr_df[correlation_features].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Ma trận tương quan của đặc trưng')
        st.pyplot(fig)
    if st.session_state.model_trained:
        st.header("Tầm quan trọng của đặc trưng trong dự đoán loại bão")
        model_results = train_model()
        with st.spinner("Tạo biểu đồ tầm quan trọng..."):
            fig = create_feature_importance_plot(model_results)
            st.plotly_chart(fig, use_container_width=True)

def show_prediction_model():
    st.title("Mô hình dự đoán loại bão")
    if not st.session_state.model_trained:
        st.info("Vui lòng huấn luyện mô hình bằng nút ở thanh bên.")
        return
    processor = st.session_state.processor
    model_results = train_model()
    st.header("Hiệu năng của mô hình")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Báo cáo phân loại")
        report = model_results['report']
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
    with col2:
        st.subheader("Ma trận nhầm lẫn")
        fig_cm = create_confusion_matrix_plot(model_results)
        st.plotly_chart(fig_cm, use_container_width=True)
    st.header("Tầm quan trọng của đặc trưng")
    fig_fi = create_feature_importance_plot(model_results)
    st.plotly_chart(fig_fi, use_container_width=True)
    st.header("Dự đoán loại bão cho quỹ đạo mới")
    uploaded_file = st.file_uploader("Tải lên file dữ liệu quỹ đạo mới (pickle hoặc CSV)", type=["pkl", "csv"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith("pkl"):
                new_data = pickle.load(uploaded_file)
            else:
                new_data = pd.read_csv(uploaded_file)
            # Giả sử new_data đã được xử lý tương tự như dữ liệu huấn luyện
            features = st.session_state.processor.model.featurizer.transform(new_data)
            prediction = st.session_state.processor.model.predict([features])[0]
            st.success(f"Dự đoán loại bão: {prediction}")
        except Exception as e:
            st.error(f"Lỗi trong quá trình dự đoán: {e}")
    st.header("Dự đoán quỹ đạo ảo từ tập kiểm tra")
    if st.session_state.data_loaded:
        dataset = st.session_state.processor.dataset if hasattr(st.session_state.processor, "dataset") else load_data()
        idx = st.number_input("Chọn số thứ tự của trajectory trong tập kiểm tra", 
                              min_value=0, max_value=len(dataset.trajs)-1, value=0, step=1)
        traj_ao = dataset.trajs[idx]
        groundtruth = dataset.labels[idx]
        try:
            # Sử dụng dữ liệu thực tế của trajectory để chuyển đổi thành vector đặc trưng
            features = st.session_state.processor.model.featurizer.transform(traj_ao)
            pred_ao = st.session_state.processor.model.predict([features])[0]
        except Exception as e:
            pred_ao = f"Lỗi: {e}"
        st.write(f"**Nhãn thực tế:** {groundtruth}")
        st.write(f"**Nhãn dự đoán:** {pred_ao}")
        fig_ao, _ = create_trajectory_map([traj_ao], [groundtruth], sample_size=1)
        st.plotly_chart(fig_ao, use_container_width=True)

def show_trajectory_comparison():
    st.title("So sánh quỹ đạo bão")
    if not st.session_state.data_loaded:
        st.info("Vui lòng load dữ liệu bão bằng nút ở thanh bên.")
        return
    processor = st.session_state.processor
    categories = sorted(processor.dataset.classes)
    selected_category = st.selectbox("Chọn loại bão để so sánh", options=["Tất cả"] + categories)
    st.header("So sánh quỹ đạo chuẩn hóa")
    with st.spinner("Tạo biểu đồ quỹ đạo chuẩn hóa..."):
        fig = create_normalized_trajectory_plot(processor, None if selected_category == "Tất cả" else selected_category)
        st.plotly_chart(fig, use_container_width=True)

def show_hurricane_impact():
    st.title("Trực quan hóa tác động bão")
    if not st.session_state.features_extracted:
        st.info("Vui lòng trích xuất đặc trưng để xem trực quan hóa tác động bão.")
        return
    processor = st.session_state.processor
    features_df = processor.features_df.copy()
    features_df = make_dataframe_arrow_compatible(features_df)
    fig = create_hurricane_impact_visualization(features_df)
    st.plotly_chart(fig, use_container_width=True)

def show_advanced_visualizations():
    st.title("Trực quan hóa nâng cao")
    processor = st.session_state.processor
    if not st.session_state.data_loaded:
        st.info("Vui lòng load dữ liệu bão.")
        return
    dataset = processor.dataset
    st.subheader("Animation Quỹ đạo bão")
    fig_map, df_points = create_trajectory_map(dataset.trajs, dataset.labels, sample_size=100)
    animated_fig = create_animated_trajectory_map(df_points)
    st.plotly_chart(animated_fig, use_container_width=True)
    st.subheader("Trực quan hóa 3D quỹ đạo bão")
    fig_3d = create_3d_trajectory_plot(dataset.trajs, dataset.labels, sample_size=20)
    st.plotly_chart(fig_3d, use_container_width=True)
    st.subheader("Biểu đồ vận tốc")
    fig_velocity = create_velocity_profile(dataset.trajs, dataset.labels, sample_size=10)
    st.plotly_chart(fig_velocity, use_container_width=True)

def show_real_input_prediction():
    st.title("Dự đoán từ dữ liệu đầu vào thực tế")
    st.write("Tải lên file CSV chứa dữ liệu trajectory với các cột: t, longitude, latitude")
    uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.subheader("Dữ liệu đầu vào:")
            st.dataframe(df_input)
            # Tạo đối tượng trajectory từ dữ liệu CSV
            traj_input = type("Trajectory", (object,), {})()
            traj_input.t = df_input["t"].values
            traj_input.r = df_input[["longitude", "latitude"]].values
            # Chuyển đổi dữ liệu trajectory thành vector đặc trưng bằng featurizer của mô hình
            features = st.session_state.processor.model.featurizer.transform(traj_input)
            prediction = st.session_state.processor.model.predict([features])[0]
            st.success(f"Dự đoán loại bão: {prediction}")
            fig, _ = create_trajectory_map([traj_input], [prediction], sample_size=1)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Lỗi xử lý dữ liệu đầu vào: {e}")

# --- Hàm chính ---
def main():
    st.sidebar.title("Phân tích bão")
    st.sidebar.header("Dữ liệu")
    if st.sidebar.button("Load Dữ liệu Bão"):
        with st.spinner("Đang load dữ liệu bão..."):
            dataset = load_data()
            st.sidebar.success(f"Đã load {len(dataset.trajs)} trajectory")
    if st.session_state.data_loaded:
        if st.sidebar.button("Trích xuất đặc trưng"):
            with st.spinner("Đang trích xuất đặc trưng..."):
                features_df = extract_features()
                st.session_state.processor.features_df = features_df
                st.sidebar.success(f"Đã trích xuất đặc trưng từ {len(features_df)} trajectory")
        if st.session_state.features_extracted and st.sidebar.button("Huấn luyện mô hình"):
            with st.spinner("Đang huấn luyện mô hình..."):
                model_results = train_model()
                st.sidebar.success(f"Mô hình đã huấn luyện với độ chính xác: {model_results['report']['accuracy']:.4f}")
    st.sidebar.header("Điều hướng")
    page = st.sidebar.radio(
        "Chọn mục",
        ["Trang chủ", "Trình duyệt quỹ đạo", "Phân tích đặc trưng", "Mô hình dự đoán", "So sánh quỹ đạo", "Trực quan hóa nâng cao", "Tác động bão", "Dữ liệu đầu vào thực tế", "Vẽ quỹ đạo để dự đoán"]
    )
    if page == "Trang chủ":
        show_home_page()
    elif page == "Trình duyệt quỹ đạo":
        show_trajectory_explorer()
    elif page == "Phân tích đặc trưng":
        show_feature_analysis()
    elif page == "Mô hình dự đoán":
        show_prediction_model()
    elif page == "So sánh quỹ đạo":
        show_trajectory_comparison()
    elif page == "Trực quan hóa nâng cao":
        show_advanced_visualizations()
    elif page == "Tác động bão":
        show_hurricane_impact()
    elif page == "Dữ liệu đầu vào thực tế":
        show_real_input_prediction()
    elif page == "Vẽ quỹ đạo để dự đoán":
        show_drawing_prediction()

if __name__ == "__main__":
    main()
