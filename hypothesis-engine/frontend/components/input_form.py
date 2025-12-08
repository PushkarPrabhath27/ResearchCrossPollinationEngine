"""
Input Form Component for Streamlit Frontend

Handles user input including query, field selection, and parameters.
Full implementation in PROMPT 21.
"""

import streamlit as st
from typing import Dict


def render_input_form() -> Dict:
    """
    Render the input form for hypothesis generation
    
    Returns:
        Dictionary with user inputs
    """
    st.header("Research Question Input")
    
    research_query = st.text_area(
        "Describe your research problem",
        placeholder="Enter a detailed description of your research question...",
        height=150,
        help="Be as specific as possible for best results"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        field = st.selectbox(
            "Primary Research Field",
            [
                "Biology",
                "Physics",
                "Chemistry",
                "Computer Science",
                "Mathematics",
                "Engineering",
                "Medicine"
            ]
        )
    
    with col2:
        num_hypotheses = st.number_input(
            "Number of Hypotheses",
            min_value=1,
            max_value=15,
            value=5
        )
    
    return {
        "query": research_query,
        "field": field.lower().replace(" ", "_"),
        "num_hypotheses": num_hypotheses
    }
