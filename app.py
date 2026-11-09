
import streamlit as st
import pandas as pd
import joblib
import json

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="StudentWell | ML Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>

.stApp {
    background: #f4f7fc;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #172554, #312e81);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #172554;
}

.subtitle {
    color: #64748b;
    font-size: 16px;
}

.hero {
    background: linear-gradient(120deg, #312e81, #2563eb, #0891b2);
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 32px;
}

.hero p {
    color: #e0f2fe;
    font-size: 16px;
}

.info-card {
    background: white;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 3px 12px rgba(0,0,0,0.04);
}

.result-card {
    background: white;
    padding: 25px;
    border-radius: 16px;
    border-left: 6px solid #2563eb;
    box-shadow: 0 4px 15px rgba(0,0,0,0.06);
}

.stButton > button {
    background: linear-gradient(90deg, #4f46e5, #2563eb);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 25px;
    font-weight: 700;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #3730a3, #1d4ed8);
    color: white;
    border: none;
}

h2, h3 {
    color: #172554;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return joblib.load("depression_model.joblib")


model = load_model()

with open("feature_info.json", "r") as file:
    info = json.load(file)


# ---------------- SIDEBAR MENU ----------------

st.sidebar.markdown("# 🎓 StudentWell")
st.sidebar.caption("Machine Learning Dashboard")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Home",
        "🔍 Prediction",
        "📊 About Model"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Powered by Random Forest ML")
st.sidebar.caption("StudentWell | 2026")


# ---------------- HOME PAGE ----------------

if page == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <h1>Welcome to StudentWell 👋</h1>
        <p>
        Explore an experimental machine-learning application
        for analysing student depression dataset patterns.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<p class="main-title">Dashboard Overview</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Your student wellness ML project at a glance.</p>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🤖 ML Algorithm", "Random Forest")

    with col2:
        st.metric("🎯 Model Accuracy", "98.21%")

    with col3:
        st.metric("📈 ROC-AUC", "99.01%")

    st.markdown("---")

    st.subheader("✨ What can you do here?")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="info-card">
            <h3>🔍 Student Prediction</h3>
            <p>Enter student information and explore
            the prediction produced by the trained model.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-card">
            <h3>📊 Model Information</h3>
            <p>Explore the algorithm, evaluation metrics
            and machine-learning workflow.</p>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "This is an educational ML demonstration. "
        "It is not a clinical diagnosis or a validated screening tool."
    )


# ---------------- PREDICTION PAGE ----------------

elif page == "🔍 Prediction":

    st.markdown("""
    <div class="hero">
        <h1>🔍 Student Prediction</h1>
        <p>Enter the required information to run your ML model.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("👤 Student Information")

    user_input = {}

    with st.form("student_prediction_form"):

        st.markdown("#### 🔢 Numerical Information")

        numeric_cols = info["numeric_columns"]

        numeric_layout = st.columns(2)

        for i, col in enumerate(numeric_cols):
            default = info["numeric_defaults"][col]

            with numeric_layout[i % 2]:
                user_input[col] = st.number_input(
                    col.replace("_", " ").title(),
                    value=float(default),
                    key=f"num_{col}"
                )

        st.markdown("---")
        st.markdown("#### 📋 Categorical Information")

        categorical_cols = info["categorical_columns"]

        categorical_layout = st.columns(2)

        for i, col in enumerate(categorical_cols):

            options = info["categorical_options"][col]

            if not options:
                continue

            with categorical_layout[i % 2]:
                user_input[col] = st.selectbox(
                    col.replace("_", " ").title(),
                    options,
                    key=f"cat_{col}"
                )

        st.markdown("---")

        submitted = st.form_submit_button(
            "🚀 Generate Prediction"
        )

    if submitted:

        input_df = pd.DataFrame([user_input])

        try:
            with st.spinner("Analysing student information..."):

                prediction = model.predict(input_df)[0]

                classes = list(
                    model.named_steps["classifier"].classes_
                )

                yes_index = classes.index("Yes")

                probability = model.predict_proba(
                    input_df
                )[0][yes_index]

            st.markdown("---")
            st.subheader("📊 Prediction Result")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicted Class",
                    str(prediction)
                )

            with col2:
                st.metric(
                    "Estimated Probability of Yes",
                    f"{probability * 100:.2f}%"
                )

            st.progress(float(probability))

            st.caption(
                "Probability displayed is the model's estimated "
                "probability for the Yes class."
            )

            st.info(
                "This experimental prediction is not a clinical "
                "diagnosis and must not be used to make healthcare decisions."
            )

        except Exception as error:
            st.error(f"Prediction error: {error}")


# ---------------- ABOUT MODEL PAGE ----------------

elif page == "📊 About Model":

    st.markdown("""
    <div class="hero">
        <h1>📊 About Our ML Model</h1>
        <p>Understanding the technology behind StudentWell.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🤖 Model Information")

    st.markdown("""
    <div class="info-card">
        <h3>Random Forest Classifier</h3>
        <p>
        Random Forest is a supervised machine-learning algorithm
        that combines multiple decision trees to perform classification.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📈 Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", "98.21%")
    col2.metric("Precision", "99.12%")
    col3.metric("Recall", "94.17%")

    col4, col5, col6 = st.columns(3)

    col4.metric("F1 Score", "96.58%")
    col5.metric("ROC-AUC", "99.01%")
    col6.metric("Training Accuracy", "100%")

    st.markdown("---")

    st.subheader("🧠 Technologies Used")

    st.write("🐍 Python")
    st.write("📚 Pandas and NumPy")
    st.write("🤖 Scikit-learn")
    st.write("🎨 Streamlit")
    st.write("💾 Joblib")

    st.warning(
        "These metrics are from the current evaluation run. "
        "They do not establish clinical validity or real-world reliability."
    )


# ---------------- FOOTER ----------------

st.markdown("---")

st.markdown(
    "<center style='color:#64748b;'>"
    "StudentWell | Educational Machine Learning Project | 2026"
    "</center>",
    unsafe_allow_html=True
)
