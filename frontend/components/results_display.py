"""
Results Display Component for Streamlit Frontend

Handles display of generated hypotheses and supporting information.
Full implementation in PROMPT 21.
"""

import streamlit as st
from typing import List, Dict


def display_hypothesis(hypothesis: Dict, index: int):
    """
    Display a single hypothesis with all details
    
    Args:
        hypothesis: Hypothesis data dictionary
        index: Hypothesis number
    """
    with st.container():
        st.markdown(f"### 💡 Hypothesis {index}: {hypothesis.get('title', 'Untitled')}")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Novelty Score", f"{hypothesis.get('novelty_score', 0):.1f}/10")
        col2.metric("Feasibility", f"{hypothesis.get('feasibility_score', 0):.1f}/10")
        col3.metric("Impact Potential", f"{hypothesis.get('impact_score', 0):.1f}/10")
        
        # Description
        st.markdown(f"**Description:** {hypothesis.get('description', 'No description')}")
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        if col1.button(f"⭐ Save", key=f"save_{index}"):
            st.success("Saved!")
        if col2.button(f"📤 Share", key=f"share_{index}"):
            st.info("Share link: [placeholder]")
        if col3.button(f"📧 Email", key=f"email_{index}"):
            st.info("Email sent! (placeholder)")


def display_results_list(hypotheses: List[Dict]):
    """
    Display a list of hypotheses
    
    Args:
        hypotheses: List of hypothesis dictionaries
    """
    st.header("Generated Hypotheses")
    
    if not hypotheses:
        st.warning("No hypotheses generated yet.")
        return
    
    for i, hypo in enumerate(hypotheses, 1):
        with st.expander(f"Hypothesis {i}: {hypo.get('title', 'Untitled')}", expanded=(i==1)):
            display_hypothesis(hypo, i)
