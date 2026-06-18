import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from openai import OpenAI

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import tensorflow as tf

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Multi-Agent Research Platform",
    layout="wide"
)

# --------------------------------------------------
# OPENAI CLIENT
# --------------------------------------------------

try:
    client = OpenAI(
        api_key=st.secrets["OPENAI_API_KEY"]
    )
except Exception:
    st.error(
        "OpenAI API key not found. Add it to .streamlit/secrets.toml"
    )
    st.stop()

# --------------------------------------------------
# AGENTS
# --------------------------------------------------

def planner_agent(goal):

    response = client.responses.create(
        model="gpt-5",
        input=f"""
        You are a Planner Agent.

        Create a detailed execution plan for:

        {goal}
        """
    )

    return response.output_text


def research_agent(query):

    response = client.responses.create(
        model="gpt-5",
        input=f"""
        You are a Research Agent.

        Research:

        {query}

        Return:
        - Key concepts
        - Recommended libraries
        - Best practices
        """
    )

    return response.output_text


def reviewer_agent(content):

    response = client.responses.create(
        model="gpt-5",
        input=f"""
        You are a Senior AI Reviewer.

        Review:

        {content}

        Provide:
        - Improvements
        - Risks
        - Suggestions
        """
    )

    return response.output_text


def executive_summary_agent(content):

    response = client.responses.create(
        model="gpt-5",
        input=f"""
        Create an executive summary:

        {content}
        """
    )

    return response.output_text

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🤖 AI Multi-Agent Research Platform")

st.markdown("""
This platform combines:

- Planner Agent
- Research Agent
- Machine Learning Agent
- CNN Agent
- Reviewer Agent
- Executive Summary Agent
""")
