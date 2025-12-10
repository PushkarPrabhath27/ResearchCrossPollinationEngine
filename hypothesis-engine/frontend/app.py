"""
Streamlit Frontend - PRODUCTION RAG UI
Displays REAL data from live APIs with working URLs

Features:
- Real paper citations with View Paper / Download PDF buttons
- Real datasets with download links
- Real GitHub repos with stars
- Comprehensive search statistics
"""

import streamlit as st
import requests
from typing import Dict, List, Any
import time

st.set_page_config(
    page_title="ScienceBridge - Research Discovery",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
.main-header { font-size: 2.5rem; color: #1E88E5; text-align: center; margin-bottom: 1rem; }
.sub-header { text-align: center; color: #666; margin-bottom: 2rem; }

.search-stats { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    color: white; padding: 1.5rem; border-radius: 12px; margin: 1rem 0; 
}
.stat-box { 
    background: rgba(255,255,255,0.2); 
    padding: 0.8rem; border-radius: 8px; text-align: center; margin: 0.5rem;
}

.paper-card { 
    background: #f8f9fa; padding: 1.2rem; border-radius: 10px; 
    margin: 0.8rem 0; border-left: 4px solid #1E88E5;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.paper-source { 
    display: inline-block; padding: 2px 8px; border-radius: 12px; 
    font-size: 0.75rem; font-weight: bold; margin-right: 8px;
}
.source-openalex { background: #4CAF50; color: white; }
.source-arxiv { background: #B31B1B; color: white; }
.source-semantic { background: #1857B6; color: white; }

.dataset-card { 
    background: #e8f5e9; padding: 1rem; border-radius: 10px; 
    margin: 0.5rem 0; border-left: 4px solid #4CAF50;
}

.repo-card { 
    background: #2d2d2d; color: #fff; padding: 1rem; border-radius: 10px; 
    margin: 0.5rem 0; font-family: monospace;
}

.btn-primary { 
    background: #1E88E5; color: white; padding: 8px 16px; 
    border-radius: 6px; text-decoration: none; display: inline-block; margin: 4px;
}
.btn-success { 
    background: #4CAF50; color: white; padding: 8px 16px; 
    border-radius: 6px; text-decoration: none; display: inline-block; margin: 4px;
}
.btn-dark { 
    background: #333; color: white; padding: 8px 16px; 
    border-radius: 6px; text-decoration: none; display: inline-block; margin: 4px;
}

.hypothesis-section { 
    background: #fff; padding: 1.5rem; border-radius: 10px; 
    box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8000"


def call_api(query: str, field: str, creativity: float) -> Dict:
    """Call the hypothesis generation API"""
    try:
        response = requests.post(
            f"{API_URL}/api/generate",
            json={
                "query": query,
                "field": field.lower().replace(" ", "_"),
                "num_hypotheses": 1,
                "creativity": creativity
            },
            timeout=180
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to backend. Is the server running?"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def display_search_stats(stats: Dict):
    """Display comprehensive search statistics"""
    if not stats:
        return
    
    st.markdown("## 🔍 Live Search Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📚 Papers Found", stats.get("total_papers_found", 0))
    with col2:
        st.metric("📊 Datasets Found", stats.get("datasets_found", 0))
    with col3:
        st.metric("💻 Repos Found", stats.get("repos_found", 0))
    with col4:
        st.metric("⏱️ Total Time", f"{stats.get('total_time_seconds', 0):.1f}s")
    
    # Source breakdown
    sources = stats.get("sources_searched", [])
    if sources:
        st.markdown("### 📡 Data Sources Searched")
        cols = st.columns(len(sources))
        for i, source in enumerate(sources):
            with cols[i]:
                st.markdown(f"""
                **{source.get('name', 'Unknown')}**
                - Available: {source.get('total_available', 0):,}
                - Returned: {source.get('returned', 0)}
                - Time: {source.get('time_seconds', 0):.2f}s
                """)


def display_real_papers(papers: List[Dict]):
    """Display REAL papers with working links"""
    if not papers:
        st.warning("No papers found for this query.")
        return
    
    st.markdown("## 📑 Real Research Papers")
    st.success(f"✅ Found {len(papers)} REAL papers with verified URLs")
    
    for i, paper in enumerate(papers[:10]):
        source = paper.get("source", "unknown")
        source_class = f"source-{source.replace('_', '-')}"
        source_icon = {"openalex": "🟢", "arxiv": "🔴", "semantic_scholar": "🔵"}.get(source, "⚪")
        
        citations = paper.get("citation_count", 0)
        pdf_url = paper.get("pdf_url", "")
        paper_url = paper.get("url", "")
        
        with st.expander(f"{source_icon} {paper.get('title', 'Untitled')[:80]}...", expanded=(i < 2)):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**📄 {paper.get('title', 'Untitled')}**")
                st.markdown(f"👥 *{paper.get('authors', 'Unknown')}*")
                st.markdown(f"📚 {paper.get('journal', 'Unknown')} ({paper.get('year', 'N/A')})")
                st.markdown(f"🔗 DOI: `{paper.get('doi', 'N/A')}`")
            
            with col2:
                st.metric("Citations", f"{citations:,}")
                st.caption(f"Source: {source}")
            
            # Abstract
            abstract = paper.get("abstract", "") or paper.get("key_finding", "")
            if abstract:
                st.markdown("**Abstract:**")
                st.markdown(f"> {abstract[:400]}{'...' if len(abstract) > 400 else ''}")
            
            # Action buttons
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                if paper_url:
                    st.markdown(f'<a href="{paper_url}" target="_blank" class="btn-primary">🔗 View Paper</a>', unsafe_allow_html=True)
            with btn_col2:
                if pdf_url:
                    st.markdown(f'<a href="{pdf_url}" target="_blank" class="btn-success">📄 Download PDF</a>', unsafe_allow_html=True)
            with btn_col3:
                doi = paper.get("doi", "")
                if doi and not doi.startswith("arXiv"):
                    st.markdown(f'<a href="https://doi.org/{doi}" target="_blank" class="btn-dark">📋 DOI Link</a>', unsafe_allow_html=True)


def display_real_datasets(datasets: List[Dict]):
    """Display REAL datasets with download links"""
    if not datasets:
        return
    
    st.markdown("## 📊 Real Datasets")
    st.success(f"✅ Found {len(datasets)} REAL datasets with download links")
    
    for dataset in datasets[:6]:
        with st.expander(f"📦 {dataset.get('name', 'Unknown')}", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Source:** {dataset.get('source', 'Unknown')}")
                st.markdown(f"**Size:** {dataset.get('size', 'Unknown')}")
                st.markdown(f"**Format:** {dataset.get('format', 'Unknown')}")
                st.markdown(f"**License:** {dataset.get('license', 'Unknown')}")
                
                desc = dataset.get("description", "")
                if desc:
                    st.markdown(f"**Description:** {desc[:200]}...")
            
            with col2:
                url = dataset.get("url", "")
                if url:
                    st.markdown(f'<a href="{url}" target="_blank" class="btn-success">📥 Access Dataset</a>', unsafe_allow_html=True)


def display_real_repos(repos: List[Dict]):
    """Display REAL GitHub repos"""
    if not repos:
        return
    
    st.markdown("## 💻 Real Code Repositories")
    st.success(f"✅ Found {len(repos)} REAL GitHub repos")
    
    for repo in repos[:5]:
        with st.expander(f"📁 {repo.get('name', 'Unknown')} ⭐ {repo.get('stars', '0')}", expanded=False):
            st.markdown(f"**Full Name:** `{repo.get('full_name', 'Unknown')}`")
            st.markdown(f"**Language:** {repo.get('language', 'Unknown')}")
            st.markdown(f"**Last Updated:** {repo.get('last_updated', 'Unknown')}")
            st.markdown(f"**License:** {repo.get('license', 'Unknown')}")
            
            desc = repo.get("description", "")
            if desc:
                st.markdown(f"**Description:** {desc}")
            
            # Clone command
            clone_url = repo.get("clone_url", "")
            if clone_url:
                st.code(clone_url, language="bash")
            
            url = repo.get("url", "")
            if url:
                st.markdown(f'<a href="{url}" target="_blank" class="btn-dark">🔗 View on GitHub</a>', unsafe_allow_html=True)


def display_hypothesis(hypothesis: Dict):
    """Display the generated hypothesis"""
    if not hypothesis:
        return
    
    st.markdown("## 💡 Generated Research Hypothesis")
    
    # Title and scores
    st.markdown(f"### {hypothesis.get('title', 'Untitled Hypothesis')}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Novelty", f"{hypothesis.get('novelty_score', 0):.1f}/10")
    with col2:
        st.metric("Feasibility", f"{hypothesis.get('feasibility_score', 0):.1f}/10")
    with col3:
        st.metric("Impact", f"{hypothesis.get('impact_score', 0):.1f}/10")
    
    # Description
    st.markdown("### 📝 Description")
    st.markdown(hypothesis.get("description", "No description"))
    
    # Theoretical Basis
    st.markdown("### 🧬 Theoretical Basis")
    st.info(hypothesis.get("theoretical_basis", "N/A"))
    
    # Why Novel
    if hypothesis.get("why_novel"):
        st.markdown("### ✨ What Makes This Novel")
        st.success(hypothesis.get("why_novel"))
    
    # Methodology Steps
    steps = hypothesis.get("methodology_steps", [])
    if steps:
        st.markdown("### 🔧 Detailed Methodology")
        for step in steps:
            with st.expander(f"Step {step.get('step_number', '?')}: {step.get('title', 'Untitled')}", expanded=False):
                st.markdown(f"**Algorithm:** {step.get('algorithm', 'N/A')}")
                st.markdown(f"**Parameters:** `{step.get('parameters', 'N/A')}`")
                st.markdown(f"**Libraries:** {', '.join(step.get('libraries', []))}")
                st.markdown(f"**Time Estimate:** {step.get('estimated_time', 'N/A')}")
                
                if step.get("source_paper"):
                    st.markdown(f"**Source Paper:** {step.get('source_paper')}")
                
                code = step.get("code_snippet", "")
                if code:
                    st.code(code, language="python")
    
    # Expected Results
    if hypothesis.get("expected_results"):
        st.markdown("### 📈 Expected Results")
        st.success(hypothesis.get("expected_results"))
    
    # Timeline
    timeline = hypothesis.get("timeline_weeks", [])
    if timeline:
        st.markdown("### ⏱️ Implementation Timeline")
        for week in timeline:
            st.markdown(f"**{week.get('week', 'Week ?')}:** {', '.join(week.get('activities', []))}")
            st.caption(f"Deliverable: {week.get('deliverable', 'N/A')}")
    
    # Budget
    if hypothesis.get("estimated_budget"):
        st.markdown("### 💰 Estimated Budget")
        st.info(hypothesis.get("estimated_budget"))
    
    # Risks
    risks = hypothesis.get("risks_and_mitigation", [])
    if risks:
        st.markdown("### ⚠️ Risks & Mitigation")
        for risk in risks:
            st.warning(f"**Risk:** {risk.get('risk', 'N/A')}")
            st.success(f"**Mitigation:** {risk.get('mitigation', 'N/A')}")
    
    # Next Steps
    next_steps = hypothesis.get("next_steps", [])
    if next_steps:
        st.markdown("### 🎯 Next Steps")
        for step in next_steps:
            st.markdown(f"- {step}")


def main():
    st.markdown('<h1 class="main-header">🔬 ScienceBridge</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Research Discovery with REAL Papers, Datasets & Code</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        field = st.selectbox(
            "Research Field",
            ["Biology", "Physics", "Chemistry", "Computer Science", 
             "Mathematics", "Engineering", "Medicine"]
        )
        
        creativity = st.slider("Creativity Level", 0.0, 1.0, 0.7)
        
        st.markdown("---")
        st.markdown("### 📡 Data Sources")
        st.markdown("""
        - 🟢 **OpenAlex** (250M+ papers)
        - 🔴 **arXiv** (2M+ papers)
        - 🔵 **Semantic Scholar** (200M+ papers)
        - 📦 **HuggingFace** (datasets)
        - 📦 **Papers With Code** (datasets)
        - 💻 **GitHub** (code repos)
        """)
        
        st.markdown("---")
        st.caption("All data is fetched LIVE from APIs. No local storage.")
    
    # Main content
    st.markdown("### 1️⃣ Describe Your Research Question")
    
    query = st.text_area(
        "What problem are you trying to solve?",
        placeholder="Example: I'm studying cancer cell migration in 3D tissue. Current imaging methods only work in 2D. What cross-disciplinary approaches from physics or engineering could help track cells in 3D?",
        height=120
    )
    
    if st.button("🚀 Generate Research Hypothesis", type="primary", use_container_width=True):
        if not query or len(query) < 20:
            st.error("Please enter a more detailed research question (at least 20 characters).")
            return
        
        with st.spinner("🔍 Searching research databases (live from APIs)..."):
            progress = st.progress(0)
            status = st.empty()
            
            steps = [
                (20, "📚 Searching OpenAlex (250M+ papers)..."),
                (40, "📚 Searching arXiv (2M+ papers)..."),
                (60, "📊 Finding relevant datasets..."),
                (80, "💻 Finding code repositories..."),
                (90, "🤖 Generating hypothesis..."),
            ]
            
            for prog, msg in steps:
                status.text(msg)
                progress.progress(prog)
                time.sleep(0.3)
            
            result = call_api(query, field, creativity)
            
            progress.progress(100)
            status.text("✅ Complete!")
        
        if result.get("success"):
            st.success("🎉 Hypothesis generated with REAL research data!")
            
            # Display search stats
            display_search_stats(result.get("search_stats", {}))
            
            st.markdown("---")
            
            # Display real papers
            display_real_papers(result.get("real_papers", []))
            
            st.markdown("---")
            
            # Display real datasets
            display_real_datasets(result.get("real_datasets", []))
            
            st.markdown("---")
            
            # Display real repos
            display_real_repos(result.get("real_repos", []))
            
            st.markdown("---")
            
            # Display generated hypothesis
            display_hypothesis(result.get("hypothesis", {}))
            
        else:
            st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
            st.info("Make sure the backend is running: `python -m uvicorn src.api.main:app --reload`")


if __name__ == "__main__":
    main()
