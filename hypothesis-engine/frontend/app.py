"""
Streamlit Frontend - Real Hypothesis Generation
"""

import streamlit as st
import requests
from typing import Dict, List
import time

st.set_page_config(
    page_title="Hypothesis Cross-Pollination Engine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1E88E5; text-align: center; margin-bottom: 2rem; }
    .hypothesis-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                       padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem; color: white; }
    .score-high { color: #4CAF50; font-weight: bold; }
    .score-med { color: #FF9800; font-weight: bold; }
    .score-low { color: #f44336; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8000"


def call_api(query: str, field: str, num_hypotheses: int, creativity: float) -> Dict:
    """Call the real hypothesis generation API"""
    try:
        response = requests.post(
            f"{API_URL}/api/generate",
            json={
                "query": query,
                "field": field.lower().replace(" ", "_"),
                "num_hypotheses": num_hypotheses,
                "creativity": creativity
            },
            timeout=120  # 2 minute timeout for LLM generation
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out. The AI is taking too long."}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to backend API. Is the server running?"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def display_score(score: float, label: str):
    """Display a score with color coding"""
    if score >= 8:
        color_class = "score-high"
    elif score >= 6:
        color_class = "score-med"
    else:
        color_class = "score-low"
    
    st.metric(label, f"{score:.1f}/10")


def main():
    st.markdown('<h1 class="main-header">🔬 Scientific Hypothesis Cross-Pollination Engine</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### Discover Novel Research Directions Through AI-Powered Cross-Disciplinary Insights
    
    This system uses **Google Gemini AI** to find innovative approaches to your research 
    problems by analyzing connections across all scientific fields.
    """)
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        field = st.selectbox(
            "Your Research Field",
            ["Biology", "Physics", "Chemistry", "Computer Science", 
             "Mathematics", "Engineering", "Medicine"]
        )
        
        num_hypotheses = st.slider("Number of Hypotheses", min_value=1, max_value=5, value=3)
        
        creativity = st.slider(
            "Creativity Level", min_value=0.0, max_value=1.0, value=0.7,
            help="Higher = more creative but riskier suggestions"
        )
        
        st.markdown("---")
        st.markdown("### 🤖 AI Status")
        
        # Check API status
        try:
            resp = requests.get(f"{API_URL}/api/stats", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                st.success(f"✅ Connected: {data.get('llm_provider', 'Unknown').upper()}")
            else:
                st.warning("⚠️ API returned error")
        except:
            st.error("❌ API not connected")
    
    # Main content
    st.header("1️⃣ Describe Your Research Question")
    
    research_query = st.text_area(
        "What problem are you trying to solve?",
        placeholder="Example: I'm studying how cancer cells migrate through blood vessels. "
                    "Current imaging techniques are limited to 2D. Are there better approaches?",
        height=150
    )
    
    if st.button("🚀 Generate Hypotheses with AI", type="primary", use_container_width=True):
        if not research_query or len(research_query) < 10:
            st.error("❌ Please enter a more detailed research question (at least 10 characters)!")
        else:
            with st.spinner("🤖 AI is generating hypotheses... This may take 30-60 seconds."):
                progress = st.progress(0)
                status = st.empty()
                
                status.text("🔍 Connecting to AI...")
                progress.progress(20)
                
                # Call the real API
                result = call_api(research_query, field, num_hypotheses, creativity)
                
                progress.progress(100)
                status.text("✅ Complete!")
            
            if result.get("success"):
                st.success("🎉 Hypotheses generated successfully!")
                display_results(result)
            else:
                st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
                st.info("💡 Make sure the backend API is running on http://localhost:8000")


def display_results(result: Dict):
    """Display the generated hypotheses"""
    
    st.markdown("---")
    st.header("💡 Generated Hypotheses")
    
    hypotheses = result.get("hypotheses", [])
    
    if not hypotheses:
        st.warning("No hypotheses were generated. Try rephrasing your question.")
        return
    
    for i, hyp in enumerate(hypotheses):
        with st.expander(f"💡 Hypothesis {i+1}: {hyp.get('title', 'Untitled')}", expanded=(i==0)):
            # Scores
            col1, col2, col3 = st.columns(3)
            with col1:
                display_score(hyp.get("novelty_score", 0), "Novelty")
            with col2:
                display_score(hyp.get("feasibility_score", 0), "Feasibility")
            with col3:
                display_score(hyp.get("impact_score", 0), "Impact")
            
            st.markdown("---")
            
            # Description
            st.markdown("**📝 Description:**")
            st.write(hyp.get("description", "No description available"))
            
            # Source fields
            if hyp.get("source_fields"):
                st.markdown("**🔗 Cross-Domain Sources:**")
                st.write(", ".join(hyp.get("source_fields", [])))
            
            # Key references
            if hyp.get("key_references"):
                st.markdown("**📚 Suggested Reading:**")
                for ref in hyp.get("key_references", []):
                    st.write(f"• {ref}")
            
            # Next steps
            if hyp.get("next_steps"):
                st.markdown("**🎯 Next Steps:**")
                for step in hyp.get("next_steps", []):
                    st.write(f"• {step}")
    
    # Cross-domain insights
    insights = result.get("cross_domain_insights", [])
    if insights:
        st.markdown("---")
        st.header("🌐 Cross-Domain Insights")
        for insight in insights:
            st.info(f"💡 {insight}")
    
    # Methodology suggestions
    methods = result.get("methodology_suggestions", [])
    if methods:
        st.markdown("---")
        st.header("🔧 Methodology Suggestions")
        for method in methods:
            st.success(f"🛠️ {method}")


if __name__ == "__main__":
    main()
