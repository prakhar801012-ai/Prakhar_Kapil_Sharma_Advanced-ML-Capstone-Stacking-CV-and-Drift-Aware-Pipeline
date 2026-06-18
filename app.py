import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from openai import OpenAI

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI + ML Multi-Agent Platform",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# OPENAI CLIENT
# --------------------------------------------------

@st.cache_resource
def get_client():
    return OpenAI(
        api_key=st.secrets["OPENAI_API_KEY"]
    )

client = get_client()

MODEL = st.secrets.get(
    "OPENAI_MODEL",
    "gpt-5"
)

# --------------------------------------------------
# AI HELPER
# --------------------------------------------------

def ask_agent(prompt):

    response = client.responses.create(
        model=MODEL,
        input=prompt
    )

    return response.output_text

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

defaults = {
    "plan": "",
    "research": "",
    "review": "",
    "summary": ""
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🤖 AI + ML Multi-Agent Platform")

st.markdown("""
This platform combines:

- Planner Agent
- Research Agent
- Machine Learning Agent
- Reviewer Agent
- Executive Summary Agent
""")

# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Planner",
    "Research",
    "ML Training",
    "Reviewer",
    "Summary"
])

# ==================================================
# PLANNER
# ==================================================

with tab1:

    st.subheader("Planner Agent")

    goal = st.text_area(
        "Describe your AI/ML project"
    )

    if st.button("Generate Plan"):

        with st.spinner("Generating plan..."):

            st.session_state.plan = ask_agent(
                f"""
                You are an AI Project Planner.

                Create a detailed roadmap for:

                {goal}

                Include:
                - Data Collection
                - Data Cleaning
                - Model Training
                - Evaluation
                - Deployment
                """
            )

    if st.session_state.plan:
        st.markdown(st.session_state.plan)

# ==================================================
# RESEARCH
# ==================================================

with tab2:

    st.subheader("Research Agent")

    topic = st.text_input(
        "Research Topic"
    )

    if st.button("Research"):

        with st.spinner("Researching..."):

            st.session_state.research = ask_agent(
                f"""
                Research:

                {topic}

                Return:

                - Key Concepts
                - Useful Libraries
                - Best Practices
                - Common Pitfalls
                """
            )

    if st.session_state.research:
        st.markdown(
            st.session_state.research
        )

# ==================================================
# ML AGENT
# ==================================================

with tab3:

    st.subheader("Machine Learning Agent")

    csv_file = st.file_uploader(
        "Upload CSV Dataset",
        type=["csv"]
    )

    if csv_file:

        df = pd.read_csv(csv_file)

        st.write("Dataset Preview")
        st.dataframe(df.head())

        target = st.selectbox(
            "Select Target Column",
            df.columns
        )

        model_type = st.radio(
            "Problem Type",
            [
                "Classification",
                "Regression"
            ]
        )

        if st.button("Train Model"):

            try:

                X = df.drop(columns=[target])
                y = df[target]

                X = pd.get_dummies(X)

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42
                )

                if model_type == "Classification":

                    model = RandomForestClassifier(
                        n_estimators=200,
                        random_state=42
                    )

                    model.fit(
                        X_train,
                        y_train
                    )

                    predictions = model.predict(
                        X_test
                    )

                    accuracy = accuracy_score(
                        y_test,
                        predictions
                    )

                    st.success(
                        f"Accuracy: {accuracy:.4f}"
                    )

                    st.text(
                        classification_report(
                            y_test,
                            predictions
                        )
                    )

                else:

                    model = RandomForestRegressor(
                        n_estimators=200,
                        random_state=42
                    )

                    model.fit(
                        X_train,
                        y_train
                    )

                    predictions = model.predict(
                        X_test
                    )

                    mse = mean_squared_error(
                        y_test,
                        predictions
                    )

                    r2 = r2_score(
                        y_test,
                        predictions
                    )

                    st.success(
                        f"R² Score: {r2:.4f}"
                    )

                    st.write(
                        f"MSE: {mse:.4f}"
                    )

                # FEATURE IMPORTANCE

                importance = pd.DataFrame({
                    "Feature": X.columns,
                    "Importance": model.feature_importances_
                })

                importance = importance.sort_values(
                    "Importance",
                    ascending=False
                ).head(15)

                st.subheader(
                    "Feature Importance"
                )

                fig, ax = plt.subplots()

                ax.barh(
                    importance["Feature"],
                    importance["Importance"]
                )

                ax.invert_yaxis()

                st.pyplot(fig)

                st.session_state.review = f"""
                Model Type: {model_type}

                Dataset Rows: {len(df)}

                Top Features:

                {importance.to_string()}
                """

            except Exception as e:

                st.error(str(e))

# ==================================================
# REVIEWER
# ==================================================

with tab4:

    st.subheader("Reviewer Agent")

    if st.button("Review ML Results"):

        if st.session_state.review:

            with st.spinner("Reviewing..."):

                review_text = ask_agent(
                    f"""
                    Review this ML project:

                    {st.session_state.review}

                    Provide:
                    - Weaknesses
                    - Improvements
                    - Suggested Algorithms
                    - Deployment Advice
                    """
                )

                st.session_state.review = review_text

    if st.session_state.review:
        st.markdown(
            st.session_state.review
        )

# ==================================================
# SUMMARY
# ==================================================

with tab5:

    st.subheader(
        "Executive Summary Agent"
    )

    findings = st.text_area(
        "Paste project findings"
    )

    if st.button(
        "Generate Executive Summary"
    ):

        with st.spinner(
            "Generating..."
        ):

            st.session_state.summary = ask_agent(
                f"""
                Create an executive summary.

                Findings:

                {findings}

                Include:
                - Objective
                - Results
                - Business Impact
                - Recommendations
                """
            )

    if st.session_state.summary:
        st.markdown(
            st.session_state.summary
        )
