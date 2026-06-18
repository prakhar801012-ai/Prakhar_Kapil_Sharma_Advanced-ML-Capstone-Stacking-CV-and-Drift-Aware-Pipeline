import streamlit as st
import pandas as pd
import numpy as np

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import tensorflow as tf

from openai import OpenAI

# --------------------------------------------------
# PAGE
# --------------------------------------------------

st.set_page_config(
    page_title="AI Multi-Agent Platform",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# OPENAI
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
# SESSION STATE
# --------------------------------------------------

for key in [
    "plan",
    "research",
    "review",
    "summary"
]:
    if key not in st.session_state:
        st.session_state[key] = ""

# --------------------------------------------------
# AGENTS
# --------------------------------------------------

def ask_agent(prompt):

    response = client.responses.create(
        model=MODEL,
        input=prompt
    )

    return response.output_text


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🤖 AI Multi-Agent Research Platform")

st.caption(
    "OpenAI + ML + CNN + Streamlit"
)

# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Planner",
    "Research",
    "Machine Learning",
    "CNN",
    "Summary"
])

# --------------------------------------------------
# PLANNER
# --------------------------------------------------

with tab1:

    st.subheader("Planner Agent")

    goal = st.text_area(
        "Project Goal"
    )

    if st.button("Generate Plan"):

        with st.spinner("Planning..."):

            st.session_state.plan = ask_agent(
                f"""
                Create a detailed project plan:

                {goal}
                """
            )

        st.success("Done")

    if st.session_state.plan:
        st.write(
            st.session_state.plan
        )

# --------------------------------------------------
# RESEARCH
# --------------------------------------------------

with tab2:

    st.subheader("Research Agent")

    topic = st.text_input(
        "Research Topic"
    )

    if st.button("Research Topic"):

        with st.spinner("Researching..."):

            st.session_state.research = ask_agent(
                f"""
                Research:

                {topic}

                Return:
                - concepts
                - libraries
                - best practices
                """
            )

    if st.session_state.research:
        st.write(
            st.session_state.research
        )

# --------------------------------------------------
# ML AGENT
# --------------------------------------------------

with tab3:

    st.subheader("Machine Learning Agent")

    csv_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if csv_file:

        df = pd.read_csv(csv_file)

        st.dataframe(
            df.head()
        )

        target = st.selectbox(
            "Target Column",
            df.columns
        )

        if st.button("Train Random Forest"):

            try:

                X = df.drop(
                    columns=[target]
                )

                X = pd.get_dummies(
                    X
                )

                y = df[target]

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42
                )

                model = RandomForestClassifier()

                model.fit(
                    X_train,
                    y_train
                )

                pred = model.predict(
                    X_test
                )

                acc = accuracy_score(
                    y_test,
                    pred
                )

                st.success(
                    f"Accuracy: {acc:.4f}"
                )

            except Exception as e:
                st.error(str(e))

# --------------------------------------------------
# CNN
# --------------------------------------------------

with tab4:

    st.subheader(
        "CNN Image Processing"
    )

    image_file = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg"]
    )

    if image_file:

        image = Image.open(
            image_file
        )

        st.image(
            image,
            use_container_width=True
        )

        img = image.resize(
            (128, 128)
        )

        img_array = np.array(
            img
        ) / 255.0

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        @st.cache_resource
        def build_cnn():

            return tf.keras.Sequential([
                tf.keras.layers.Input(
                    shape=(128,128,3)
                ),

                tf.keras.layers.Conv2D(
                    32,
                    3,
                    activation="relu"
                ),

                tf.keras.layers.MaxPool2D(),

                tf.keras.layers.Conv2D(
                    64,
                    3,
                    activation="relu"
                ),

                tf.keras.layers.MaxPool2D(),

                tf.keras.layers.Flatten(),

                tf.keras.layers.Dense(
                    128,
                    activation="relu"
                ),

                tf.keras.layers.Dense(
                    2,
                    activation="softmax"
                )
            ])

        cnn_model = build_cnn()

        st.success(
            "CNN model loaded."
        )

        st.code(
            cnn_model.summary(
                print_fn=lambda x: x
            )
        )

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

with tab5:

    st.subheader(
        "Executive Summary Agent"
    )

    findings = st.text_area(
        "Paste Findings"
    )

    if st.button(
        "Generate Summary"
    ):

        with st.spinner(
            "Generating..."
        ):

            st.session_state.summary = ask_agent(
                f"""
                Create executive summary:

                {findings}
                """
            )

    if st.session_state.summary:
        st.write(
            st.session_state.summary
        )
