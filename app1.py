"""
PG Accommodation Analysis: Clustering & Classification
Streamlit Dashboard
-------------------------------------------------------
Run with:
    streamlit run app.py

Expected files in the same folder (optional but recommended):
    Final_PG_Dataset.csv
    preprocessor.pkl          (joblib dump from your notebook)
    kmeans_model.pkl          (joblib dump from your notebook)
    cluster_classifier.pkl    (joblib dump from your notebook)

If the .pkl artifacts are not found, the app will train everything
live from the CSV so it still works end-to-end.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA

try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except ImportError:
    HAS_OPTION_MENU = False

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="PG Analysis",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "Final_PG_Dataset.csv"
PREPROCESSOR_PATH = "preprocessor.pkl"
KMEANS_PATH = "kmeans_model.pkl"
CLASSIFIER_PATH = "cluster_classifier.pkl"

DEFAULT_CLUSTER_FEATURES = [
    "Monthly_Rent", "Distance_to_IT_Park_km", "Distance_to_Metro_km",
    "Safety_Score", "Commute_Score", "Value_for_Money", "Google_Rating",
]

# ------------------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------------------
st.markdown("""
<style>
    .block-container {padding-top: 1.5rem;}
    .kpi-card {
        border-radius: 12px;
        padding: 18px 20px;
        color: white;
        height: 108px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .kpi-title {font-size: 13px; opacity: 0.9; margin-bottom: 4px;}
    .kpi-value {font-size: 30px; font-weight: 700; line-height: 1.1;}
    .kpi-sub {font-size: 12px; opacity: 0.85; margin-top: 2px;}
    .workflow-card {
        border-radius: 10px;
        padding: 16px 18px;
        height: 190px;
        border: 1px solid rgba(0,0,0,0.06);
    }
    .workflow-title {font-weight: 700; font-size: 15px; margin-bottom: 10px;}
    .info-banner {
        background-color: #EAF2FE;
        border-left: 5px solid #2E7BF6;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 14px;
        color: #1a3a5c;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df


@st.cache_resource
def load_artifact(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


if not os.path.exists(DATA_PATH):
    st.error(f"Could not find `{DATA_PATH}`. Place it in the same folder as app.py.")
    st.stop()

df = load_data(DATA_PATH)
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# session state
if "cluster_labels" not in st.session_state:
    st.session_state.cluster_labels = None
if "cluster_features" not in st.session_state:
    st.session_state.cluster_features = DEFAULT_CLUSTER_FEATURES.copy()
if "kmeans_model" not in st.session_state:
    st.session_state.kmeans_model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None


# ------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏠 PG Analysis")
    st.caption("Navigation")

    top_level = ["Dashboard", "Dataset Overview", "Clustering", "Classification", "About"]
    top_icons = ["speedometer2", "table", "diagram-3", "bar-chart", "info-circle"]

    if HAS_OPTION_MENU:
        selected = option_menu(
            menu_title=None,
            options=top_level,
            icons=top_icons,
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"font-size": "15px"},
                "nav-link": {"font-size": "14px", "text-align": "left", "margin": "2px"},
                "nav-link-selected": {"background-color": "#2E7BF6"},
            },
        )
    else:
        selected = st.radio("Go to", top_level, label_visibility="collapsed")

    st.markdown("---")
    st.caption(f"Rows loaded: {len(df):,}")


# ------------------------------------------------------------------
# PAGE: DASHBOARD  (matches the reference screenshot)
# ------------------------------------------------------------------
def page_dashboard():
    st.markdown(
        "<h1 style='margin-bottom:0px;'>🏠 PG Accommodation Analysis: Clustering & Classification</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Unsupervised Learning → Supervised Learning")
    st.write("")

    total_pgs = len(df)
    n_numeric = len(numeric_cols)
    n_categorical = len(categorical_cols)
    missing_vals = int(df.isnull().sum().sum())
    dup_rows = int(df.duplicated().sum())

    kpi_data = [
        ("Total PGs", f"{total_pgs:,}", "Records", "#2E7BF6"),
        ("Numerical Features", f"{n_numeric}", "Features", "#2FA36B"),
        ("Categorical Features", f"{n_categorical}", "Features", "#E0932C"),
        ("Missing Values", f"{missing_vals}", "Entries", "#8B6BD1"),
        ("Duplicate Rows", f"{dup_rows}", "Rows", "#D9534F"),
    ]

    cols = st.columns(5)
    for c, (title, value, sub, color) in zip(cols, kpi_data):
        c.markdown(f"""
        <div class="kpi-card" style="background-color:{color};">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    col1, col2, col3 = st.columns(3)

    # --- Occupancy Status Donut ---
    with col1:
        st.markdown("**Occupancy Status Distribution**")
        occ_counts = df["Occupancy_Status"].value_counts().reset_index()
        occ_counts.columns = ["Status", "Count"]
        fig = px.pie(
            occ_counts, names="Status", values="Count", hole=0.55,
            color_discrete_sequence=["#2E7BF6", "#2FA36B", "#E0932C", "#D9534F"],
        )
        fig.update_traces(textposition="inside", textinfo="percent")
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320,
                           legend=dict(orientation="v", y=0.5))
        st.plotly_chart(fig, use_container_width=True)

    # --- Average Rent by Room Type ---
    with col2:
        st.markdown("**Average Rent by Room Type**")
        rent_by_room = (
            df.groupby("Room_Type")["Monthly_Rent"].mean().sort_values(ascending=False).reset_index()
        )
        fig = px.bar(
            rent_by_room, x="Room_Type", y="Monthly_Rent",
            color="Room_Type",
            color_discrete_sequence=["#2E7BF6", "#2FA36B", "#E0932C", "#D9534F", "#8B6BD1"],
        )
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), height=320,
                           xaxis_title="", yaxis_title="Average Rent (₹)")
        st.plotly_chart(fig, use_container_width=True)

    # --- PGs by Locality Top 10 ---
    with col3:
        st.markdown("**PGs by Locality (Top 10)**")
        top_locality = df["Area"].value_counts().head(10).sort_values(ascending=True).reset_index()
        top_locality.columns = ["Locality", "Count"]
        fig = px.bar(
            top_locality, x="Count", y="Locality", orientation="h",
            color_discrete_sequence=["#2E7BF6"],
        )
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320,
                           xaxis_title="Number of PGs", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.write("")
    st.markdown("**Workflow Overview**")

    wcols = st.columns([1, 0.15, 1, 0.15, 1, 0.15, 1])
    workflow_steps = [
        ("1. Data Preprocessing", "#2E7BF6", ["Handle missing values", "Encode categorical features", "Feature scaling"]),
        ("2. Clustering (Unsupervised)", "#2FA36B", ["K-Means Clustering", "Find optimal number of clusters", "Analyze cluster characteristics"]),
        ("3. Classification (Supervised)", "#E0932C", ["Train multiple models", "Compare performance", "Select best model"]),
        ("4. Results & Insights", "#8B6BD1", ["Model evaluation", "Confusion matrix", "Business insights"]),
    ]
    positions = [0, 2, 4, 6]
    for pos, (title, color, items) in zip(positions, workflow_steps):
        with wcols[pos]:
            bullets = "".join([f"<li>{it}</li>" for it in items])
            st.markdown(f"""
            <div class="workflow-card" style="border-top: 4px solid {color};">
                <div class="workflow-title" style="color:{color};">{title}</div>
                <ul style="padding-left:18px; font-size:13px; line-height:1.6;">{bullets}</ul>
            </div>
            """, unsafe_allow_html=True)
    for pos in [1, 3, 5]:
        with wcols[pos]:
            st.markdown("<div style='height:190px; display:flex; align-items:center; justify-content:center; font-size:22px;'>➜</div>",
                        unsafe_allow_html=True)

    st.write("")
    st.markdown("""
    <div class="info-banner">
        ℹ️ Follow the workflow from left to right. Start with dataset overview, perform clustering, then build classification models.
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------
# PAGE: DATASET OVERVIEW
# ------------------------------------------------------------------
def page_dataset_overview():
    st.header("📋 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Column Data Types")
    dtypes_df = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str).values,
        "Unique Values": [df[c].nunique() for c in df.columns],
        "Missing": df.isnull().sum().values,
    })
    st.dataframe(dtypes_df, use_container_width=True, height=350)

    st.subheader("Numerical Summary")
    st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

    st.subheader("Correlation Heatmap (Numerical Features)")
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------
# CLUSTERING SUB-PAGES
# ------------------------------------------------------------------
def clustering_feature_selection():
    st.subheader("🎯 Feature Selection")
    st.write("Choose the numerical features used to build the clusters.")
    selected = st.multiselect(
        "Features for clustering", numeric_cols,
        default=[f for f in st.session_state.cluster_features if f in numeric_cols],
    )
    if selected:
        st.session_state.cluster_features = selected
        fig = px.imshow(df[selected].corr(), color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         text_auto=".2f", aspect="auto")
        fig.update_layout(height=450, title="Correlation among selected features")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Select at least 2 features to continue.")


def get_scaled_data():
    feats = st.session_state.cluster_features
    X = df[feats].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    st.session_state.scaler = scaler
    return X, X_scaled


def clustering_elbow_method():
    st.subheader("📉 Elbow Method")
    if len(st.session_state.cluster_features) < 2:
        st.warning("Select at least 2 features in 'Feature Selection' first.")
        return

    max_k = st.slider("Max number of clusters (k) to test", 2, 15, 10)
    _, X_scaled = get_scaled_data()

    wcss = []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        wcss.append(km.inertia_)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, max_k + 1)), y=wcss, mode="lines+markers",
                              line=dict(color="#2E7BF6", width=3)))
    fig.update_layout(title="WCSS vs Number of Clusters", xaxis_title="k",
                       yaxis_title="Within-Cluster Sum of Squares", height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.info("Look for the 'elbow' point where WCSS stops decreasing sharply — that's a good candidate for k.")


def clustering_silhouette_score():
    st.subheader("📊 Silhouette Score")
    if len(st.session_state.cluster_features) < 2:
        st.warning("Select at least 2 features in 'Feature Selection' first.")
        return

    max_k = st.slider("Max number of clusters (k) to test", 2, 12, 8, key="sil_max_k")
    _, X_scaled = get_scaled_data()

    scores = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        scores.append(silhouette_score(X_scaled, labels))

    best_k = list(range(2, max_k + 1))[int(np.argmax(scores))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(2, max_k + 1)), y=scores, mode="lines+markers",
                              line=dict(color="#2FA36B", width=3)))
    fig.update_layout(title="Silhouette Score vs Number of Clusters", xaxis_title="k",
                       yaxis_title="Silhouette Score", height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"Best silhouette score at **k = {best_k}**")


def clustering_kmeans():
    st.subheader("🧩 K-Means Clustering")
    if len(st.session_state.cluster_features) < 2:
        st.warning("Select at least 2 features in 'Feature Selection' first.")
        return

    k = st.slider("Number of clusters (k)", 2, 10, 3, key="kmeans_k")
    X, X_scaled = get_scaled_data()

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    st.session_state.kmeans_model = km
    result_df = df.loc[X.index].copy()
    result_df["Cluster"] = labels
    st.session_state.cluster_labels = result_df["Cluster"]
    st.session_state.clustered_df = result_df

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Cluster Sizes**")
        counts = result_df["Cluster"].value_counts().sort_index().reset_index()
        counts.columns = ["Cluster", "Count"]
        counts["Cluster"] = counts["Cluster"].astype(str)
        fig = px.bar(counts, x="Cluster", y="Count",
                     color="Cluster", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Clusters visualized (PCA 2D projection)**")
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)
        plot_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
        plot_df["Cluster"] = labels.astype(str)
        fig = px.scatter(plot_df, x="PC1", y="PC2", color="Cluster",
                          color_discrete_sequence=px.colors.qualitative.Set2, opacity=0.7)
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.success(f"K-Means fit complete with k = {k}. Labels saved for use in the Classification section.")


def clustering_cluster_profiling():
    st.subheader("🔍 Cluster Profiling")
    if st.session_state.cluster_labels is None:
        st.warning("Run 'K-Means Clustering' first to generate cluster labels.")
        return

    result_df = st.session_state.clustered_df
    feats = st.session_state.cluster_features

    st.markdown("**Average feature values per cluster**")
    profile = result_df.groupby("Cluster")[feats].mean().round(2)
    st.dataframe(profile, use_container_width=True)

    fig = px.imshow(profile.T, color_continuous_scale="Blues", aspect="auto",
                     text_auto=".1f", labels=dict(color="Value"))
    fig.update_layout(height=420, title="Cluster Feature Heatmap")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Occupancy Status by Cluster**")
    occ_ct = pd.crosstab(result_df["Cluster"], result_df["Occupancy_Status"])
    fig2 = px.bar(occ_ct, barmode="group", height=380)
    st.plotly_chart(fig2, use_container_width=True)


def page_clustering():
    st.markdown("## 🧬 Clustering (Unsupervised Learning)")
    tabs = st.tabs(["Feature Selection", "Elbow Method", "Silhouette Score",
                     "K-Means Clustering", "Cluster Profiling"])
    with tabs[0]:
        clustering_feature_selection()
    with tabs[1]:
        clustering_elbow_method()
    with tabs[2]:
        clustering_silhouette_score()
    with tabs[3]:
        clustering_kmeans()
    with tabs[4]:
        clustering_cluster_profiling()


# ------------------------------------------------------------------
# CLASSIFICATION SUB-PAGES
# ------------------------------------------------------------------
MODEL_ZOO = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "SVM": SVC(probability=True, random_state=42),
}


def classification_model_training():
    st.subheader("🏋️ Model Training")
    if st.session_state.cluster_labels is None:
        st.warning("Run K-Means Clustering first — the classifier learns to predict cluster membership.")
        return

    result_df = st.session_state.clustered_df
    feats = st.session_state.cluster_features
    X = result_df[feats]
    y = result_df["Cluster"]

    test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
    chosen_models = st.multiselect("Models to train", list(MODEL_ZOO.keys()),
                                    default=list(MODEL_ZOO.keys()))

    if st.button("Train Models", type="primary"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        trained = {}
        rows = []
        progress = st.progress(0)
        for i, name in enumerate(chosen_models):
            model = MODEL_ZOO[name]
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
            acc = accuracy_score(y_test, preds)
            trained[name] = model
            rows.append({"Model": name, "Accuracy": round(acc, 4)})
            progress.progress((i + 1) / len(chosen_models))

        st.session_state.trained_models = trained
        st.session_state.X_test = X_test_s
        st.session_state.y_test = y_test
        st.session_state.class_scaler = scaler

        st.success("Training complete!")
        st.dataframe(pd.DataFrame(rows).sort_values("Accuracy", ascending=False),
                     use_container_width=True)


def classification_model_comparison():
    st.subheader("⚖️ Model Comparison")
    if not st.session_state.trained_models:
        st.warning("Train models first in the 'Model Training' tab.")
        return

    rows = []
    for name, model in st.session_state.trained_models.items():
        preds = model.predict(st.session_state.X_test)
        acc = accuracy_score(st.session_state.y_test, preds)
        rows.append({"Model": name, "Accuracy": acc})
    comp_df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)

    fig = px.bar(comp_df, x="Model", y="Accuracy", color="Model", text_auto=".3f",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(showlegend=False, height=420, yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(comp_df, use_container_width=True)


def classification_confusion_matrix():
    st.subheader("🧮 Confusion Matrix")
    if not st.session_state.trained_models:
        st.warning("Train models first in the 'Model Training' tab.")
        return

    model_name = st.selectbox("Select model", list(st.session_state.trained_models.keys()))
    model = st.session_state.trained_models[model_name]
    preds = model.predict(st.session_state.X_test)
    cm = confusion_matrix(st.session_state.y_test, preds)

    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                     labels=dict(x="Predicted", y="Actual", color="Count"))
    fig.update_layout(height=450, title=f"Confusion Matrix — {model_name}")
    st.plotly_chart(fig, use_container_width=True)


def classification_report_page():
    st.subheader("📄 Classification Report")
    if not st.session_state.trained_models:
        st.warning("Train models first in the 'Model Training' tab.")
        return

    model_name = st.selectbox("Select model", list(st.session_state.trained_models.keys()), key="report_model")
    model = st.session_state.trained_models[model_name]
    preds = model.predict(st.session_state.X_test)
    report = classification_report(st.session_state.y_test, preds, output_dict=True)
    report_df = pd.DataFrame(report).T.round(3)
    st.dataframe(report_df, use_container_width=True)


def classification_predictions():
    st.subheader("🔮 Predictions on New Data")
    feats = st.session_state.cluster_features
    if not feats:
        st.warning("Select clustering features first.")
        return

    st.write("Enter feature values to predict which cluster a new PG would belong to.")
    input_vals = {}
    cols = st.columns(3)
    for i, f in enumerate(feats):
        with cols[i % 3]:
            default_val = float(df[f].median())
            input_vals[f] = st.number_input(f, value=default_val)

    if st.button("Predict Cluster", type="primary"):
        model_choice = None
        if st.session_state.trained_models:
            model_choice = st.session_state.get("pred_model_name") or list(st.session_state.trained_models.keys())[0]
            model = st.session_state.trained_models[model_choice]
            scaler = st.session_state.get("class_scaler")
            X_new = pd.DataFrame([input_vals])[feats]
            X_new_s = scaler.transform(X_new)
            pred = model.predict(X_new_s)[0]
            st.success(f"Predicted Cluster: **{pred}**  (using {model_choice})")
        elif st.session_state.kmeans_model is not None:
            X_new = pd.DataFrame([input_vals])[feats]
            X_new_s = st.session_state.scaler.transform(X_new)
            pred = st.session_state.kmeans_model.predict(X_new_s)[0]
            st.success(f"Predicted Cluster (direct K-Means assignment): **{pred}**")
        else:
            st.error("No trained model or K-Means fit available yet. Complete the Clustering and Model Training steps first.")


def page_classification():
    st.markdown("## 🧠 Classification (Supervised Learning)")
    tabs = st.tabs(["Model Training", "Model Comparison", "Confusion Matrix",
                     "Classification Report", "Predictions"])
    with tabs[0]:
        classification_model_training()
    with tabs[1]:
        classification_model_comparison()
    with tabs[2]:
        classification_confusion_matrix()
    with tabs[3]:
        classification_report_page()
    with tabs[4]:
        classification_predictions()


# ------------------------------------------------------------------
# PAGE: ABOUT
# ------------------------------------------------------------------
def page_about():
    st.header("ℹ️ About this Project")
    st.markdown("""
    **PG Accommodation Analysis: Clustering & Classification**

    This dashboard analyzes PG (Paying Guest) accommodation listings across Bengaluru,
    combining **unsupervised learning** (K-Means clustering) with **supervised learning**
    (multi-model classification) to segment properties and predict cluster membership
    for new listings.

    **Workflow**
    1. Data Preprocessing — handle missing values, encode categorical features, scale numerical features
    2. Clustering — K-Means to discover natural groupings of PGs based on price, location, and amenity scores
    3. Classification — train and compare multiple classifiers to predict cluster membership
    4. Results & Insights — confusion matrices, classification reports, and business takeaways

    Built with **Streamlit**, **scikit-learn**, and **Plotly**.
    """)


# ------------------------------------------------------------------
# ROUTER
# ------------------------------------------------------------------
if selected == "Dashboard":
    page_dashboard()
elif selected == "Dataset Overview":
    page_dataset_overview()
elif selected == "Clustering":
    page_clustering()
elif selected == "Classification":
    page_classification()
elif selected == "About":
    page_about()