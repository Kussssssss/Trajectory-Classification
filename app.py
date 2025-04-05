import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
import pickle

from data_processing import HurricaneDataProcessor

# Cấu hình trang
st.set_page_config(
    page_title="Phân tích quỹ đạo bão và dự đoán",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Khởi tạo session state
if 'processor' not in st.session_state:
    st.session_state.processor = HurricaneDataProcessor()
    st.session_state.data_loaded = False
    st.session_state.features_extracted = False
    st.session_state.model_trained = False
    st.session_state.selected_trajectory = None

# Hàm load dữ liệu
@st.cache_resource
def load_data():
    processor = st.session_state.processor
    dataset = processor.load_data()
    st.session_state.data_loaded = True
    return dataset

# Hàm trích xuất đặc trưng
@st.cache_data
def extract_features():
    processor = st.session_state.processor
    features_df = processor.extract_features()
    st.session_state.features_extracted = True
    return features_df

# Hàm huấn luyện mô hình
@st.cache_resource
def train_model():
    processor = st.session_state.processor
    model_results = processor.train_model()
    st.session_state.model_trained = True
    # Lưu mô hình để sử dụng sau
    processor.save_model()
    return model_results

# Hàm tạo bản đồ quỹ đạo sử dụng Plotly
def create_trajectory_map(trajectories, labels, sample_size=50):
    # Lấy mẫu nếu số lượng quá lớn
    if len(trajectories) > sample_size:
        indices = np.random.choice(len(trajectories), sample_size, replace=False)
        sample_trajs = [trajectories[i] for i in indices]
        sample_labels = [labels[i] for i in indices]
    else:
        sample_trajs = trajectories
        sample_labels = labels
    
    # Tạo dataframe để vẽ
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
    
    # Tạo bản đồ với Plotly Express
    fig = px.line_geo(
        df, 
        lat='latitude', 
        lon='longitude',
        color='category',
        color_discrete_sequence=['blue', 'green', 'red', 'purple', 'orange', 'brown'],
        line_group='traj_id',
        title='Quỹ đạo bão theo loại'
    )
    
    # Thêm điểm khởi đầu
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
        legend_title_text='Loại bão'
    )
    
    return fig

# Hàm tạo biểu đồ tầm quan trọng của đặc trưng sử dụng Seaborn
def create_feature_importance_plot(model_results):
    feature_importance = model_results['feature_importance']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10), ax=ax)
    ax.set_title('Top 10 Tầm quan trọng của đặc trưng cho dự đoán loại bão')
    ax.set_xlabel('Tầm quan trọng')
    ax.set_ylabel('Đặc trưng')
    
    return fig

# Hàm tạo biểu đồ ma trận nhầm lẫn sử dụng Seaborn
def create_confusion_matrix_plot(model_results):
    cm = model_results['confusion_matrix']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=sorted(set(model_results['y_test'])),
                yticklabels=sorted(set(model_results['y_test'])))
    ax.set_xlabel('Dự đoán')
    ax.set_ylabel('Thật')
    ax.set_title('Ma trận nhầm lẫn')
    
    return fig

# Hàm tạo biểu đồ phân phối đặc trưng
def create_feature_distribution_plot(features_df, feature_name):
    # Sao chép dữ liệu và chuyển đổi giá trị nếu cần (ví dụ: chuyển vector thành độ lớn)
    df = features_df.copy()
    
    def convert_vector_to_norm(x):
        try:
            if isinstance(x, (list, np.ndarray)):
                return np.linalg.norm(x)
            if hasattr(x, 'norm'):
                return x.norm
        except Exception as e:
            return x
        return x

    df[feature_name] = df[feature_name].apply(convert_vector_to_norm)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='category', y=feature_name, data=df,
                palette=['blue', 'green', 'red', 'purple', 'orange', 'brown'],
                ax=ax)
    ax.set_title(f'Phân phối {feature_name} theo loại bão')
    ax.set_xlabel('Loại bão')
    ax.set_ylabel(feature_name)
    
    return fig

# Hàm tạo biểu đồ quỹ đạo chuẩn hóa để so sánh
def create_normalized_trajectory_plot(processor, category=None):
    # Lấy mẫu quỹ đạo (10 mỗi loại)
    samples = processor.get_sample_trajectories(n_per_category=10)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, cat in enumerate(sorted(samples.keys())):
        ax = axes[i]
        # Nếu lọc theo loại và không khớp thì bỏ qua
        if category is not None and cat != category:
            continue
            
        for traj in samples[cat]:
            if len(traj) >= 3:  # chỉ vẽ các trajectory có đủ điểm
                r_norm = processor.normalize_trajectory(traj)
                ax.plot(r_norm[:, 0], r_norm[:, 1],
                        color=processor.get_category_color(cat), alpha=0.5)
                ax.scatter(0, 0, color='black', s=20)  # Đánh dấu gốc tọa độ
        
        ax.set_title(f'Loại {cat}: Quỹ đạo chuẩn hóa')
        ax.set_xlabel('X chuẩn hóa')
        ax.set_ylabel('Y chuẩn hóa')
        ax.grid(True)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig

# Trang chủ
def show_home_page():
    st.title("Phân tích quỹ đạo bão và dự đoán")
    st.write("""
    Chào mừng bạn đến với ứng dụng phân tích quỹ đạo bão và dự đoán loại bão. Dashboard tương tác này cho phép bạn khám phá dữ liệu quỹ đạo bão,
    trực quan hóa các mẫu, và dự đoán loại bão dựa trên các đặc trưng quỹ đạo.
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
            st.plotly_chart(fig)
    
    st.header("Các mục trong ứng dụng")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Trình duyệt quỹ đạo")
        st.write("Trực quan hóa quỹ đạo bão trên bản đồ tương tác và khám phá phân bố địa lý theo loại.")
        st.subheader("Phân tích đặc trưng")
        st.write("Phân tích các đặc trưng chính (vận tốc, độ dài, thời lượng, v.v.) và khám phá mối tương quan.")
    with col2:
        st.subheader("Mô hình dự đoán")
        st.write("Dự đoán loại bão dựa trên các đặc trưng quỹ đạo và đánh giá hiệu năng mô hình.")
        st.subheader("So sánh quỹ đạo")
        st.write("So sánh các quỹ đạo chuẩn hóa giữa các loại bão khác nhau.")

# Trang Trình duyệt quỹ đạo
def show_trajectory_explorer():
    st.title("Trình duyệt quỹ đạo bão")
    
    if not st.session_state.data_loaded:
        st.info("Vui lòng load dữ liệu bão bằng nút ở thanh bên.")
        return
    
    processor = st.session_state.processor
    
    # Bộ lọc ở thanh bên
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
    
    # Lọc trajectory theo loại
    filtered_indices = [i for i, label in enumerate(processor.dataset.labels) if label in selected_categories]
    filtered_trajs = [processor.dataset.trajs[i] for i in filtered_indices]
    filtered_labels = [processor.dataset.labels[i] for i in filtered_indices]
    
    st.write(f"Hiển thị {min(sample_size, len(filtered_trajs))} trajectory trên tổng số {len(filtered_trajs)} trajectory đã lọc.")
    
    with st.spinner("Tạo bản đồ quỹ đạo..."):
        fig = create_trajectory_map(filtered_trajs, filtered_labels, sample_size)
        st.plotly_chart(fig, use_container_width=True)
    
    st.header("Thống kê trajectory theo loại")
    if st.session_state.features_extracted:
        features_df = processor.features_df
        filtered_features = features_df[features_df['category'].isin(selected_categories)]
        grouped = filtered_features.groupby('category').agg({
            'traj_length': ['mean', 'min', 'max'],
            'traj_duration': ['mean', 'min', 'max'],
            'mean_velocity': ['mean', 'min', 'max'],
            'lon_range': ['mean', 'min', 'max'],
            'lat_range': ['mean', 'min', 'max']
        }).reset_index()
        grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
        st.dataframe(grouped)
    else:
        st.info("Vui lòng trích xuất đặc trưng bằng nút ở thanh bên để xem thống kê trajectory.")

# Trang Phân tích đặc trưng
def show_feature_analysis():
    st.title("Phân tích đặc trưng bão")
    
    if not st.session_state.features_extracted:
        st.info("Vui lòng trích xuất đặc trưng bằng nút ở thanh bên.")
        return
    
    processor = st.session_state.processor
    features_df = processor.features_df
    
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
        st.pyplot(fig)
    
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
        model_results = train_model()  # cached result
        with st.spinner("Tạo biểu đồ tầm quan trọng của đặc trưng..."):
            fig = create_feature_importance_plot(model_results)
            st.pyplot(fig)

# Trang Mô hình dự đoán
def show_prediction_model():
    st.title("Mô hình dự đoán loại bão")
    
    if not st.session_state.model_trained:
        st.info("Vui lòng huấn luyện mô hình bằng nút ở thanh bên.")
        return
    
    processor = st.session_state.processor
    model_results = train_model()  # cached result
    
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
        st.pyplot(fig_cm)
    
    st.header("Tầm quan trọng của đặc trưng")
    fig_fi = create_feature_importance_plot(model_results)
    st.pyplot(fig_fi)
    
    st.header("Dự đoán loại bão cho quỹ đạo mới")
    uploaded_file = st.file_uploader("Tải lên file dữ liệu quỹ đạo mới (pickle hoặc CSV)", type=["pkl", "csv"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith("pkl"):
                new_data = pickle.load(uploaded_file)
            else:
                new_data = pd.read_csv(uploaded_file)
            # Giả sử processor có hàm predict_new_trajectory để xử lý dữ liệu mới
            prediction = processor.predict_new_trajectory(new_data)
            st.success(f"Dự đoán loại bão: {prediction}")
        except Exception as e:
            st.error(f"Lỗi trong quá trình dự đoán: {e}")
    
    st.header("Dự đoán quỹ đạo ảo từ tập kiểm tra")
    # Nếu dữ liệu đã load, cho phép chọn một trajectory từ tập kiểm tra để tạo input ảo
    if st.session_state.data_loaded:
        processor = st.session_state.processor
        # Giả sử dataset đã được load và lưu trong processor
        dataset = processor.dataset if hasattr(processor, "dataset") else load_data()
        idx = st.number_input("Chọn số thứ tự của trajectory trong tập kiểm tra", min_value=0, max_value=len(dataset.trajs)-1, value=0, step=1)
        traj_ao = dataset.trajs[idx]
        groundtruth = dataset.labels[idx]
        try:
            pred_ao = processor.predict_new_trajectory(traj_ao)
        except Exception as e:
            pred_ao = f"Lỗi: {e}"
        st.write(f"**Nhãn thực tế:** {groundtruth}")
        st.write(f"**Nhãn dự đoán:** {pred_ao}")
        # Visualize trajectory ảo
        fig_ao = create_trajectory_map([traj_ao], [groundtruth], sample_size=1)
        st.plotly_chart(fig_ao, use_container_width=True)

# Trang So sánh quỹ đạo
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
        st.pyplot(fig)

# Hàm chính
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
        ["Trang chủ", "Trình duyệt quỹ đạo", "Phân tích đặc trưng", "Mô hình dự đoán", "So sánh quỹ đạo"]
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

if __name__ == "__main__":
    main()
