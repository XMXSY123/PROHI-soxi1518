import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
CONFIG = {
    'THRESHOLD_SEPSIS': 0.7,
    'THRESHOLD_UNSURE': 0.4,
    'MODEL_PATH': 'models/xgboost_model.pkl',
    'LOGO_PATH': './assets/project-logo.jpg',
    'PARLIAMENT_SIZE': 15,
    'CONSENSUS_STRONG': 80,
    'CONSENSUS_MODERATE': 60,
    'TOP_FEATURES_COUNT': 10
}

FEATURE_COLUMNS = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
    'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
    'Fibrinogen', 'Platelets', 'Age', 'Gender', 'HospAdmTime'
]

FEATURE_CATEGORIES = {
    "Vital Signs": ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2'],
    "Blood Gas": ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2'],
    "Organ Function": ['AST', 'BUN', 'Alkalinephos', 'Creatinine'],
    "Metabolic": ['Calcium', 'Chloride', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium'],
    "Hematology": ['Bilirubin_direct', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets'],
    "Demographics": ['Age', 'Gender', 'HospAdmTime']
}

NORMAL_RANGES = {
    'HR': (60, 100), 'O2Sat': (95, 100), 'Temp': (36.1, 37.2), 'SBP': (90, 140), 
    'MAP': (70, 100), 'DBP': (60, 90), 'Resp': (12, 20), 'pH': (7.35, 7.45),
    'Glucose': (70, 140), 'Lactate': (0.5, 2.2), 'WBC': (4.0, 11.0), 'Hct': (38, 52),
    'Hgb': (12, 18), 'Platelets': (150, 450), 'Age': (18, 100)
}

st.set_page_config(
    page_title="PROHI Sepsis Prediction - Parliament of Doctors",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    color: #2E86AB;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    font-weight: bold;
    color: #A23B72;
    margin: 1rem 0;
}
.doctor-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem;
    text-align: center;
    border-left: 5px solid #2E86AB;
}
.sepsis-positive {
    background-color: #ffebee;
    border-left-color: #f44336;
}
.sepsis-negative {
    background-color: #e8f5e8;
    border-left-color: #4caf50;
}
.sepsis-unsure {
    background-color: #fff3e0;
    border-left-color: #ff9800;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model = joblib.load(CONFIG['MODEL_PATH'])
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def create_advisory_board_visualization(tree_predictions, tree_scores):
    n_trees = len(tree_predictions)
    
    decisions = []
    colors = []
    
    for i, (pred, score) in enumerate(zip(tree_predictions, tree_scores)):
        if pred == 1:
            decisions.append("Sepsis")
            colors.append("#f44336")
        else:
            decisions.append("No Sepsis")
            colors.append("#4caf50")
    
    sepsis_count = decisions.count("Sepsis")
    no_sepsis_count = decisions.count("No Sepsis")
    
    angles = np.linspace(0, np.pi, n_trees)
    x = np.cos(angles)
    y = np.sin(angles)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(
            size=CONFIG['PARLIAMENT_SIZE'],
            color=colors,
            line=dict(width=2, color='white'),
            symbol='circle'
        ),
        text=[f"AI Advisor {i+1}<br>Decision: {dec}<br>Score: {score:.4f}<br>Vote: {'Sepsis' if pred == 1 else 'No Sepsis'}" 
              for i, (dec, pred, score) in enumerate(zip(decisions, tree_predictions, tree_scores))],
        hovertemplate='%{text}<extra></extra>',
        showlegend=False
    ))
    
    fig.update_layout(
        title=dict(
            text=f"AI Advisory Board ({n_trees} XGBoost Trees)<br>" +
                 f"<span style='color:#f44336'>Sepsis: {sepsis_count}</span> | " +
                 f"<span style='color:#4caf50'>No Sepsis: {no_sepsis_count}</span>",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        width=800,
        height=400,
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    return fig, sepsis_count, no_sepsis_count

def get_individual_tree_predictions(model, X):
    tree_predictions = []
    tree_scores = []
    
    # Convert to DMatrix for XGBoost
    if hasattr(model, 'get_booster'):  # XGBoost model
        dmatrix = xgb.DMatrix(X)
        
        # Get predictions from each tree (boosting rounds)
        for i in range(model.n_estimators):
            # Get prediction from trees up to iteration i
            tree_score = model.get_booster().predict(dmatrix, iteration_range=(i, i+1))[0]
            tree_scores.append(tree_score)
            
            # Convert score to binary prediction (XGBoost uses logits)
            tree_pred = 1 if tree_score > 0 else 0
            tree_predictions.append(tree_pred)
    
    else:  # Fallback for Random Forest or other models
        for tree in model.estimators_:
            pred = tree.predict(X)[0]
            prob = tree.predict_proba(X)[0]
            tree_predictions.append(pred)
            tree_scores.append(prob[1])  # Use probability as score
    
    return tree_predictions, tree_scores

def main():
    st.markdown('<h1 class="main-header">🩺 PROHI Sepsis Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">AI Advisory Board - XGBoost Ensemble for Sepsis Detection</h2>', unsafe_allow_html=True)
    
    st.sidebar.image(CONFIG['LOGO_PATH'], width=200)
    st.sidebar.markdown("## About")
    st.sidebar.info("""
    This dashboard uses an XGBoost model where each tree acts as an "AI advisor" 
    contributing to sepsis diagnosis. The advisory board visualization shows how 
    individual trees vote and their continuous scores combine for the final prediction.
    """)
    
    model = load_model()
    if model is None:
        st.error("Could not load the trained model. Please ensure the model file exists.")
        return
    
    st.success(f"✅ XGBoost model loaded successfully! ({model.n_estimators} trees ready to advise)")
    
    st.markdown('<h3 class="sub-header">📋 Patient Information Input</h3>', unsafe_allow_html=True)
    
    input_data = {}
    
    tabs = st.tabs(list(FEATURE_CATEGORIES.keys()))
    
    for i, (category, features) in enumerate(FEATURE_CATEGORIES.items()):
        with tabs[i]:
            st.markdown(f"**{category}**")
            
            cols = st.columns(3)
            
            for j, feature in enumerate(features):
                col_idx = j % 3
                
                with cols[col_idx]:
                    if feature in NORMAL_RANGES:
                        min_val, max_val = NORMAL_RANGES[feature]
                        help_text = f"Normal range: {min_val}-{max_val}"
                    else:
                        min_val, max_val = 0.0, 1000.0
                        help_text = "Enter the measured value"
                    
                    if feature == 'Gender':
                        value = st.selectbox(f"{feature}", options=[0, 1], 
                                           format_func=lambda x: "Female" if x == 0 else "Male",
                                           help="0=Female, 1=Male")
                    elif feature in ['Unit1', 'Unit2']:
                        value = st.selectbox(f"{feature}", options=[0, 1], 
                                           format_func=lambda x: "No" if x == 0 else "Yes",
                                           help=f"Is patient in {feature}?")
                    else:
                        if feature in NORMAL_RANGES:
                            default_val = (min_val + max_val) / 2
                        else:
                            default_val = 0.0
                        
                        value = st.number_input(
                            f"{feature}", 
                            min_value=float(min_val), 
                            max_value=float(max_val),
                            value=float(default_val),
                            step=0.1 if max_val <= 100 else 1.0,
                            help=help_text
                        )
                    
                    input_data[feature] = value
    
    if st.button("🔍 Get Doctor Opinions", type="primary", use_container_width=True):
        X_input = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
        X_input = X_input.fillna(0)
        
        prediction = model.predict(X_input)[0]
        probability = model.predict_proba(X_input)[0]
        
        tree_preds, tree_scores = get_individual_tree_predictions(model, X_input)
        
        st.markdown("---")
        st.markdown('<h3 class="sub-header">🏛️ AI Advisory Board Decision</h3>', unsafe_allow_html=True)
        
        overall_prob = probability[1]
        if overall_prob >= 0.5:
            st.error(f"🚨 **SEPSIS RISK DETECTED** (Confidence: {overall_prob:.1%})")
        else:
            st.success(f"✅ **LOW SEPSIS RISK** (Confidence: {1-overall_prob:.1%})")
        
        fig, sepsis_votes, no_sepsis_votes = create_advisory_board_visualization(
            tree_preds, tree_scores
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="doctor-card sepsis-positive">
                <h4>🔴 Sepsis</h4>
                <h2>{sepsis_votes}</h2>
                <p>advisors</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="doctor-card sepsis-negative">
                <h4>🟢 No Sepsis</h4>
                <h2>{no_sepsis_votes}</h2>
                <p>advisors</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<h3 class="sub-header">📊 Detailed Analysis</h3>', unsafe_allow_html=True)
        
        score_df = pd.DataFrame({
            'Advisor': [f"Tree {i+1}" for i in range(len(tree_scores))],
            'Tree_Score': tree_scores,
            'Decision': ["Sepsis" if pred == 1 else "No Sepsis" for pred in tree_preds]
        })
        
        fig_hist = px.histogram(
            score_df, 
            x='Tree_Score', 
            color='Decision',
            nbins=20,
            title="Distribution of XGBoost Tree Scores",
            color_discrete_map={
                'Sepsis': '#f44336',
                'No Sepsis': '#4caf50'
            }
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="gray", 
                          annotation_text="Decision Boundary (Score = 0)")
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': FEATURE_COLUMNS,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(CONFIG['TOP_FEATURES_COUNT'])
            
            fig_importance = px.bar(
                importance_df, 
                x='Importance', 
                y='Feature',
                orientation='h',
                title=f"Top {CONFIG['TOP_FEATURES_COUNT']} Most Important Features"
            )
            fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
            
            st.plotly_chart(fig_importance, use_container_width=True)
        
        st.markdown("---")
        st.markdown('<h3 class="sub-header">🩺 Clinical Interpretation</h3>', unsafe_allow_html=True)
        
        consensus_pct = max(sepsis_votes, no_sepsis_votes) / len(tree_preds) * 100
        
        if consensus_pct >= CONFIG['CONSENSUS_STRONG']:
            consensus_level = "Strong"
            consensus_color = "green" if no_sepsis_votes > sepsis_votes else "red"
        elif consensus_pct >= CONFIG['CONSENSUS_MODERATE']:
            consensus_level = "Moderate"
            consensus_color = "orange"
        else:
            consensus_level = "Weak"
            consensus_color = "gray"
        
        st.markdown(f"""
        **Consensus Level:** <span style="color:{consensus_color}">**{consensus_level}**</span> ({consensus_pct:.1f}% agreement)
        
        **XGBoost Tree Summary:**
        - Total trees voting Sepsis: {sepsis_votes}
        - Total trees voting No Sepsis: {no_sepsis_votes}
        - Average tree score: {np.mean(tree_scores):.4f}
        - Final prediction confidence: {overall_prob:.1%}
        
        **Recommendation:**
        """, unsafe_allow_html=True)
        
        if sepsis_votes > no_sepsis_votes:
            if consensus_pct >= 70:
                st.error("🚨 **HIGH PRIORITY**: Strong XGBoost consensus for sepsis risk. Immediate clinical evaluation recommended.")
            else:
                st.warning("⚠️ **MODERATE PRIORITY**: XGBoost indicates sepsis risk. Close monitoring and further evaluation advised.")
        elif no_sepsis_votes > sepsis_votes:
            if consensus_pct >= 70:
                st.success("✅ **LOW PRIORITY**: Strong XGBoost consensus for low sepsis risk. Continue routine monitoring.")
            else:
                st.info("ℹ️ **ROUTINE MONITORING**: XGBoost indicates low sepsis risk but continue standard care protocols.")
        else:
            st.warning("🤔 **SPLIT DECISION**: Equal XGBoost tree votes. Consider additional clinical assessment and laboratory tests.")

if __name__ == "__main__":
    main()