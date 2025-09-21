import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="PROHI Dashboard",
    page_icon="",
    layout="wide"
)

st.sidebar.image("./assets/img.png", width=200)
st.title("PROHI Dashboard")
st.markdown("---")


st.subheader("Dashboard Controls")
col1, col2, col3 = st.columns(3)


with col1:
    st.write("**Patient Category**")
    patient_category = st.selectbox(
        "Select patient type:",
        ["Inpatient", "Outpatient", "Emergency", "ICU"],
        key="patient_cat"
    )

# 输入小部件 2: 滑块
with col2:
    st.write("**Age Range**")
    age_range = st.slider(
        "Select age range:",
        0, 100, (25, 65),
        key="age_slider"
    )

# 输入小部件 3: 日期选择器
with col3:
    st.write("**Analysis Period**")
    date_range = st.date_input(
        "Select date range:",
        value=[pd.Timestamp('2025-01-01'), pd.Timestamp('2025-09-20')],
        key="date_range"
    )

st.markdown("---")

# 数据展示部分
st.subheader("Data Overview")

# 生成合成数据
np.random.seed(42)
data = {
    'Patient_ID': [f'P{str(i).zfill(4)}' for i in range(1, 101)],
    'Age': np.random.randint(18, 80, 100),
    'Gender': np.random.choice(['Male', 'Female'], 100),
    'Department': np.random.choice(['Cardiology', 'Neurology', 'Orthopedics', 'Emergency'], 100),
    'Length_of_Stay': np.random.randint(1, 15, 100),
    'Treatment_Cost': np.random.normal(5000, 2000, 100).astype(int),
    'Satisfaction_Score': np.random.uniform(3.0, 5.0, 100).round(1)
}

df = pd.DataFrame(data)
df['Treatment_Cost'] = df['Treatment_Cost'].clip(lower=1000)  # 确保费用为正数

# 显示数据表格
st.dataframe(
    df.head(20),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")


st.subheader("Data Visualizations")


chart_col1, chart_col2 = st.columns(2)


with chart_col1:
    st.write("**Patient Distribution by Department**")
    dept_counts = df['Department'].value_counts()

    fig_pie = px.pie(
        values=dept_counts.values,
        names=dept_counts.index,
        title="Patients by Department",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

with chart_col2:
    st.write("**Age Distribution of Patients**")
    fig_hist = px.histogram(
        df,
        x='Age',
        nbins=20,
        title="Patient Age Distribution",
        color_discrete_sequence=['#636EFA']
    )
    fig_hist.update_layout(
        xaxis_title="Age",
        yaxis_title="Number of Patients"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

st.write("**Treatment Cost vs Length of Stay**")
fig_scatter = px.scatter(
    df,
    x='Length_of_Stay',
    y='Treatment_Cost',
    color='Department',
    size='Satisfaction_Score',
    title="Treatment Cost vs Length of Stay by Department",
    hover_data=['Patient_ID', 'Age', 'Gender']
)
fig_scatter.update_layout(
    xaxis_title="Length of Stay (days)",
    yaxis_title="Treatment Cost ($)"
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")


# 页脚
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>PROHI Dashboard | Health Informatics Analytics Platform</p>
    </div>
    """,
    unsafe_allow_html=True
)

