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


def display_cross_domain(cross_domain: Dict):
    """Display cross-domain search results and connections"""
    if not cross_domain or not cross_domain.get("cross_domain_results"):
        return
    
    st.markdown("## 🌐 Cross-Domain Discoveries")
    st.info("💡 **The Core Innovation**: Finding connections between different fields!")
    
    results = cross_domain.get("cross_domain_results", [])
    connections = cross_domain.get("connections", [])
    
    # Display connections first
    if connections:
        st.markdown("### 🔗 Cross-Domain Connections Found")
        for conn in connections[:3]:
            with st.expander(f"🔄 {conn.get('from_field', 'Unknown')} → {conn.get('to_field', 'Unknown')}", expanded=True):
                st.markdown(f"**Reasoning:** {conn.get('reasoning', 'N/A')}")
                
                key_paper = conn.get("key_paper", {})
                if key_paper:
                    st.markdown(f"""
                    **Key Paper:** "{key_paper.get('title', 'Untitled')}"  
                    - Authors: {key_paper.get('authors', 'Unknown')}
                    - Year: {key_paper.get('year', 'N/A')}, Citations: {key_paper.get('citations', 0)}
                    """)
                
                st.success(f"**Potential Application:** {conn.get('potential_application', 'N/A')}")
    
    # Display papers from each related field
    for result in results[:2]:
        field_name = result.get("field", "Unknown").replace("_", " ").title()
        papers = result.get("papers", [])
        
        if papers:
            st.markdown(f"### 📚 Papers from {field_name}")
            st.caption(f"Search reasoning: {result.get('reasoning', 'N/A')}")
            
            for paper in papers[:3]:
                with st.container():
                    st.markdown(f"""
                    <div class="paper-card">
                        <strong>{paper.get('title', 'Untitled')}</strong><br>
                        <em>{paper.get('authors', 'Unknown')} ({paper.get('year', 'N/A')})</em><br>
                        <small>Citations: {paper.get('citation_count', 0)}</small>
                    </div>
                    """, unsafe_allow_html=True)


def display_real_experts(experts: List[Dict]):
    """Display REAL experts with their profiles"""
    if not experts:
        return
    
    st.markdown("## 👨‍🔬 Real Expert Recommendations")
    st.success(f"✅ Found {len(experts)} REAL researchers from OpenAlex")
    
    for expert in experts[:5]:
        potential = expert.get("collaboration_potential", "MEDIUM")
        potential_color = {"HIGH": "🟢", "MEDIUM": "🟡", "ACCESSIBLE": "🟠"}.get(potential, "⚪")
        
        with st.expander(f"{potential_color} {expert.get('name', 'Unknown')} - {expert.get('affiliation', 'Unknown')}", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Affiliation:** {expert.get('affiliation', 'Unknown')}")
                st.markdown(f"**Country:** {expert.get('country', 'Unknown')}")
                st.markdown(f"**Citations:** {expert.get('citations', 'N/A')}")
                st.markdown(f"**h-index:** {expert.get('h_index', 'N/A')}")
                st.markdown(f"**Works Count:** {expert.get('works_count', 'N/A')}")
                
                topics = expert.get("research_topics", [])
                if topics:
                    st.markdown(f"**Research Topics:** {', '.join(topics[:5])}")
            
            with col2:
                st.markdown(f"**Collaboration Potential:** {potential}")
                
                orcid_url = expert.get("orcid_url")
                if orcid_url:
                    st.markdown(f'<a href="{orcid_url}" target="_blank" class="btn-success">🔗 ORCID Profile</a>', unsafe_allow_html=True)
                
                openalex_url = expert.get("openalex_url")
                if openalex_url:
                    st.markdown(f'<a href="{openalex_url}" target="_blank" class="btn-primary">📚 OpenAlex Profile</a>', unsafe_allow_html=True)


def display_quality_score(quality: Dict):
    """Display quality validation score (X/10 for each dimension)"""
    if not quality:
        return
    
    st.markdown("## 📊 Hypothesis Quality Score")
    
    # Overall score with color
    overall = quality.get("overall", {})
    score = overall.get("score", 0)
    status = overall.get("status", "⚠️")
    
    if score >= 7:
        st.success(f"### {status} Overall Quality: {score}/10 - Good!")
    elif score >= 5:
        st.warning(f"### {status} Overall Quality: {score}/10 - Needs Improvement")
    else:
        st.error(f"### {status} Overall Quality: {score}/10 - Major Issues")
    
    # Individual metrics
    col1, col2, col3, col4 = st.columns(4)
    
    citation = quality.get("citation_quality", {})
    with col1:
        st.metric(
            "📚 Citation Quality", 
            f"{citation.get('score', 0)}/10",
            delta=None
        )
        if citation.get("status") == "❌":
            st.caption("⚠️ Fabricated citations found!")
    
    specificity = quality.get("specificity", {})
    with col2:
        st.metric("🔢 Specificity", f"{specificity.get('score', 0)}/10")
        if specificity.get("score", 0) < 5:
            st.caption("Needs more numbers")
    
    cross_domain = quality.get("cross_domain", {})
    with col3:
        st.metric("🔄 Cross-Domain", f"{cross_domain.get('score', 0)}/10")
    
    actionability = quality.get("actionability", {})
    with col4:
        st.metric("✅ Actionability", f"{actionability.get('score', 0)}/10")
    
    # Issues
    issues = quality.get("issues", [])
    if issues:
        with st.expander("⚠️ Quality Issues Found", expanded=False):
            for issue in issues[:10]:
                st.warning(issue)
    
    # Fabricated citations
    fabricated = quality.get("fabricated_citations", [])
    if fabricated:
        with st.expander("❌ Fabricated Citations Detected", expanded=True):
            st.error("The following citations were NOT found in retrieved papers:")
            for cite in fabricated[:5]:
                st.markdown(f"- `{cite}`")


def display_problem_context(context: Dict):
    """Display problem context: SOTA, failed attempts, unmet need"""
    if not context:
        return
    
    st.markdown("## 🔍 Problem Context & Literature Gap")
    
    # Current SOTA
    sota = context.get("current_sota", {})
    if sota:
        st.markdown("### 📈 Current State of the Art")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**Method:** {sota.get('method', 'N/A')}")
            st.markdown(f"**Performance:** {sota.get('performance', 'N/A')}")
            st.markdown(f"**Source:** {sota.get('source', 'N/A')}")
        with col2:
            st.warning(f"**Limitation:** {sota.get('limitation', 'N/A')}")
    
    # Failed attempts
    failed = context.get("failed_attempts", [])
    if failed:
        st.markdown("### ❌ What Has Been Tried (and Failed)")
        for attempt in failed[:3]:
            with st.expander(f"🔴 {attempt.get('approach', 'Unknown approach')}", expanded=False):
                st.markdown(f"**Researchers:** {attempt.get('researchers', 'N/A')}")
                st.markdown(f"**Result:** {attempt.get('result', 'N/A')}")
                st.markdown(f"**Why Failed:** {attempt.get('why_failed', 'N/A')}")
                st.caption(f"Source: {attempt.get('source', 'N/A')}")
    
    # Unmet need
    unmet = context.get("unmet_need", "")
    if unmet:
        st.markdown("### 🎯 The Gap We're Filling")
        st.info(unmet)


def display_comparison_table(comparison: Dict):
    """Display comparison table with existing methods"""
    if not comparison:
        return
    
    st.markdown("## 📊 Comparison with Existing Methods")
    
    methods = comparison.get("methods", [])
    if methods:
        # Create table data
        table_data = []
        for method in methods:
            table_data.append({
                "Method": method.get("name", ""),
                "Performance": method.get("performance", ""),
                "Cost": method.get("cost", ""),
                "Advantages": ", ".join(method.get("advantages", [])[:2]),
                "Limitations": ", ".join(method.get("limitations", [])[:2])
            })
        
        if table_data:
            import pandas as pd
            df = pd.DataFrame(table_data)
            st.table(df)
    
    recommendation = comparison.get("recommendation", "")
    if recommendation:
        st.success(f"**Recommendation:** {recommendation}")


def display_methodology_new(methodology: list):
    """Display detailed methodology steps with new format"""
    if not methodology:
        return
    
    st.markdown("## 🔧 Detailed Methodology")
    
    for step in methodology:
        step_num = step.get("step_number", "?")
        step_name = step.get("step_name", "Untitled Step")
        
        with st.expander(f"📍 Step {step_num}: {step_name}", expanded=(step_num == 1)):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Algorithm:** `{step.get('algorithm', 'N/A')}`")
                
                # Parameters
                params = step.get("parameters", {})
                if params:
                    st.markdown("**Parameters:**")
                    for param, value in params.items():
                        st.markdown(f"- `{param}`: {value}")
                
                # Source papers
                sources = step.get("source_papers", [])
                if sources:
                    st.markdown("**Source Papers:**")
                    for src in sources[:3]:
                        st.markdown(f"- {src}")
            
            with col2:
                st.markdown(f"**Time Estimate:** {step.get('time_estimate', 'N/A')}")
                st.markdown(f"**Resources:** {step.get('resources_needed', 'N/A')}")
                st.markdown(f"**Success Criteria:** {step.get('success_criteria', 'N/A')}")
            
            # Input/Output
            st.markdown(f"**Input:** {step.get('input_spec', 'N/A')}")
            st.markdown(f"**Output:** {step.get('output_spec', 'N/A')}")
            
            # Code snippet
            code = step.get("code_snippet")
            if code:
                st.code(code, language="python")


def display_validation_metrics(metrics: Dict):
    """Display validation success metrics"""
    if not metrics:
        return
    
    st.markdown("## ✅ Success Metrics")
    
    primary = metrics.get("primary_metrics", [])
    if primary:
        st.markdown("### Primary Metrics")
        for metric in primary[:3]:
            with st.container():
                st.markdown(f"**{metric.get('metric', 'Unknown')}**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"Current SOTA: `{metric.get('current_sota', 'N/A')}`")
                with col2:
                    st.markdown(f"Our Target: `{metric.get('your_target', 'N/A')}`")
                with col3:
                    st.markdown(f"Success Threshold: `{metric.get('success_threshold', 'N/A')}`")
                st.caption(f"Measurement: {metric.get('measurement_method', 'N/A')}")
                st.markdown("---")


def display_risk_assessment(risks: List[Dict]):
    """Display risk assessment with probabilities"""
    if not risks:
        return
    
    st.markdown("## ⚠️ Risk Assessment")
    
    for risk in risks[:3]:
        impact = risk.get("impact", "MEDIUM")
        color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(impact, "⚪")
        
        with st.expander(f"{color} {risk.get('risk', 'Unknown Risk')}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Probability:** {risk.get('probability', 'N/A')}")
                st.markdown(f"**Impact:** {impact}")
                st.markdown(f"**Evidence:** {risk.get('evidence', 'N/A')}")
            with col2:
                st.success(f"**Mitigation:** {risk.get('mitigation', 'N/A')}")
                st.info(f"**Contingency:** {risk.get('contingency', 'N/A')}")


def display_new_hypothesis(h: Dict):
    """Display hypothesis in new format from updatesprompt.md schema"""
    if not h:
        return
    
    # Title and executive summary
    title = h.get("hypothesis_title") or h.get("title", "Untitled Hypothesis")
    st.markdown(f"## 💡 {title}")
    
    # NEW: Enhanced executive summary
    exec_new = h.get("executive_summary_new", {})
    if exec_new:
        st.info(f"**{exec_new.get('one_sentence', 'N/A')}**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"🎯 **Audience:** {exec_new.get('target_audience', 'N/A')}")
        with col2:
            st.markdown(f"✨ **Innovation:** {exec_new.get('key_innovation', 'N/A')}")
    else:
        executive = h.get("executive_summary")
        if executive:
            st.info(f"**Executive Summary:** {executive}")
    
    # Main hypothesis
    hyp = h.get("hypothesis", {})
    if hyp:
        st.markdown("### 🎯 Main Claim")
        st.markdown(hyp.get("main_claim", h.get("description", "N/A")))
        
        if hyp.get("theoretical_basis"):
            st.markdown("### 🧬 Theoretical Basis")
            st.markdown(hyp.get("theoretical_basis"))
        
        if hyp.get("novelty"):
            st.markdown("### ✨ What's Novel")
            st.success(hyp.get("novelty"))
        
        if hyp.get("expected_improvement"):
            st.markdown("### 📈 Expected Improvement")
            st.success(hyp.get("expected_improvement"))
    else:
        # Fallback to old format
        if h.get("description"):
            st.markdown("### 📝 Description")
            st.markdown(h.get("description"))
        
        if h.get("theoretical_basis"):
            st.markdown("### 🧬 Theoretical Basis")
            st.info(h.get("theoretical_basis"))


def display_novelty_analysis(novelty: Dict):
    """Display novelty analysis - what's new and why possible now"""
    if not novelty:
        return
    
    st.markdown("## 🆕 Novelty Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### What Has NOT Been Done")
        st.info(novelty.get("what_has_not_been_done", "N/A"))
        
        st.markdown("### Why Not Done Before")
        st.warning(novelty.get("why_not_done_before", "N/A"))
    
    with col2:
        st.markdown("### Why Possible Now")
        st.success(novelty.get("why_possible_now", "N/A"))
        
        lit_search = novelty.get("literature_search", {})
        if lit_search:
            st.markdown("### Literature Search")
            st.markdown(f"**Query:** {lit_search.get('query_used', 'N/A')}")
            st.markdown(f"**Papers Found:** {lit_search.get('papers_found', 0)}")
            st.markdown(f"**Closest Work:** {lit_search.get('closest_work', 'N/A')}")


def display_why_not_done_before(why: Dict):
    """Display Section A: Why This Hasn't Been Done Before"""
    if not why:
        return
    
    st.markdown("## 🤔 Why This Hasn't Been Done Before")
    
    barriers = [
        ("⏰ Temporal Barrier", why.get("temporal_barrier")),
        ("🔧 Technical Barrier", why.get("technical_barrier")),
        ("🏛️ Cultural Barrier", why.get("cultural_barrier")),
        ("📊 Data Barrier", why.get("data_barrier"))
    ]
    
    col1, col2 = st.columns(2)
    for i, (label, value) in enumerate(barriers):
        with col1 if i % 2 == 0 else col2:
            if value:
                st.markdown(f"**{label}**")
                st.markdown(value)
    
    if why.get("opportunity_now"):
        st.success(f"**🎯 The Opportunity is NOW:** {why.get('opportunity_now')}")


def display_alternatives_rejected(alternatives: Dict):
    """Display Section B: Alternative Approaches Rejected"""
    if not alternatives:
        return
    
    st.markdown("## 🔀 Alternative Approaches Considered (& Rejected)")
    
    alts = alternatives.get("alternatives", [])
    for alt in alts[:4]:
        with st.expander(f"❌ {alt.get('option_name', 'Alternative')}", expanded=False):
            st.markdown(f"**What is it:** {alt.get('what_is_it', 'N/A')}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Pros:**")
                for pro in alt.get("pros", []):
                    st.markdown(f"✅ {pro}")
            with col2:
                st.markdown("**Cons:**")
                for con in alt.get("cons", []):
                    st.markdown(f"❌ {con}")
            
            st.error(f"**Why Rejected:** {alt.get('why_rejected', 'N/A')}")
    
    if alternatives.get("why_our_approach_best"):
        st.success(f"**✅ Why Our Approach is Best:** {alternatives.get('why_our_approach_best')}")


def display_preliminary_data(prelim: Dict):
    """Display Section C: Preliminary Data / Proof of Concept"""
    if not prelim:
        return
    
    st.markdown("## 🧪 Preliminary Data / Proof of Concept")
    
    pilots = prelim.get("pilot_studies", [])
    for pilot in pilots[:3]:
        with st.expander(f"✅ {pilot.get('study_name', 'Pilot Study')}", expanded=True):
            st.markdown(f"**Date:** {pilot.get('date', 'N/A')}")
            st.markdown(f"**Method:** {pilot.get('method', 'N/A')}")
            
            results = pilot.get("results", [])
            if results:
                st.markdown("**Results:**")
                for r in results:
                    st.markdown(f"- {r}")
            
            st.info(f"**Conclusion:** {pilot.get('conclusion', 'N/A')}")
            st.markdown(f"**Next Steps:** {pilot.get('next_steps', 'N/A')}")
    
    if prelim.get("overall_readiness"):
        st.success(f"**📊 Overall Readiness:** {prelim.get('overall_readiness')}")


def display_broader_impact(impact: Dict):
    """Display Section D: Broader Impact"""
    if not impact:
        return
    
    st.markdown("## 🌍 Broader Impact")
    
    impact_areas = [
        ("🏥 Clinical", impact.get("clinical_impact")),
        ("💰 Economic", impact.get("economic_impact")),
        ("🔬 Scientific", impact.get("scientific_impact")),
        ("🎓 Educational", impact.get("educational_impact")),
        ("🌱 Environmental", impact.get("environmental_impact"))
    ]
    
    cols = st.columns(3)
    for i, (label, area) in enumerate(impact_areas):
        if area:
            with cols[i % 3]:
                st.markdown(f"**{label}**")
                st.markdown(area.get("description", "N/A"))
                st.success(f"📊 {area.get('quantitative_benefit', 'N/A')}")
    
    sdgs = impact.get("sdg_alignment", [])
    if sdgs:
        st.markdown("**UN SDG Alignment:**")
        st.markdown(" · ".join(sdgs))


def display_funding_opportunities(funding: Dict):
    """Display Section E: Funding Opportunities"""
    if not funding:
        return
    
    st.markdown("## 💰 Funding Opportunities")
    
    opps = funding.get("opportunities", [])
    for opp in opps[:4]:
        fit = opp.get("fit_score", 0)
        emoji = "🎯" if fit >= 8 else "👍" if fit >= 6 else "🤔"
        
        with st.expander(f"{emoji} {opp.get('agency', 'N/A')} - {opp.get('program', 'N/A')}", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Amount", opp.get("amount", "N/A"))
            with col2:
                st.metric("Duration", opp.get("duration", "N/A"))
            with col3:
                st.metric("Fit Score", f"{fit}/10")
            
            st.markdown(f"**Deadline:** {opp.get('deadline', 'N/A')}")
            st.markdown(f"**Success Rate:** {opp.get('success_rate', 'N/A')}")
            
            fits = opp.get("why_good_fit", [])
            if fits:
                st.markdown("**Why Good Fit:**")
                for f in fits:
                    st.markdown(f"✅ {f}")
            
            st.info(f"**Strategy:** {opp.get('application_strategy', 'N/A')}")
    
    if funding.get("application_timeline"):
        st.markdown(f"**📅 Timeline:** {funding.get('application_timeline')}")
    
    if funding.get("recommended_first"):
        st.success(f"**💡 Recommendation:** {funding.get('recommended_first')}")


def display_ip_landscape(ip: Dict):
    """Display Section F: IP Landscape"""
    if not ip:
        return
    
    st.markdown("## 🔒 Intellectual Property Landscape")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Patents Found", ip.get("patents_found", 0))
        terms = ip.get("search_terms", [])
        if terms:
            st.markdown(f"**Search Terms:** {', '.join(terms)}")
    
    with col2:
        st.info(f"**Recommendation:** {ip.get('recommendation', 'N/A')}")
        st.markdown(f"**Estimated Cost:** {ip.get('estimated_cost', 'N/A')}")
    
    patents = ip.get("relevant_patents", [])
    if patents:
        st.markdown("### Relevant Patents")
        for patent in patents[:3]:
            risk = patent.get("risk_level", "MEDIUM")
            color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(risk, "⚪")
            
            with st.expander(f"{color} {patent.get('patent_number', 'N/A')} - {patent.get('title', 'N/A')}", expanded=False):
                st.markdown(f"**Assignee:** {patent.get('assignee', 'N/A')}")
                st.markdown(f"**Filed:** {patent.get('filed_year', 'N/A')}, Status: {patent.get('status', 'N/A')}")
                st.markdown(f"**Potential Conflict:** {patent.get('potential_conflict', 'N/A')}")
                st.markdown(f"**Risk Reasoning:** {patent.get('risk_reasoning', 'N/A')}")
                st.success(f"**Mitigation:** {patent.get('mitigation', 'N/A')}")


def display_enhanced_experts(experts: List[Dict]):
    """Display enhanced experts with email templates"""
    if not experts:
        return
    
    st.markdown("## 👨‍🔬 Expert Collaborators (with Contact Templates)")
    
    for exp in experts[:3]:
        priority = exp.get("priority", "SECONDARY")
        emoji = "🔴" if priority == "HIGHEST" else "🟡" if priority == "SECONDARY" else "🟢"
        
        with st.expander(f"{emoji} {exp.get('name', 'Unknown')} - {exp.get('institution', 'Unknown')}", expanded=priority=="HIGHEST"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Email:** {exp.get('email', 'N/A')}")
                st.markdown(f"**Expertise:** {exp.get('expertise_summary', 'N/A')}")
                
                papers = exp.get("relevant_papers", [])
                if papers:
                    st.markdown("**Relevant Papers:**")
                    for p in papers[:3]:
                        st.markdown(f"- {p}")
            
            with col2:
                likelihood = exp.get("collaboration_likelihood", {})
                if likelihood:
                    st.markdown(f"**Likelihood:** {likelihood.get('likelihood', 'N/A')}")
                    
                    evidence_for = likelihood.get("evidence_for", [])
                    if evidence_for:
                        st.markdown("*Why might accept:*")
                        for e in evidence_for[:2]:
                            st.markdown(f"✅ {e}")
                    
                    evidence_against = likelihood.get("evidence_against", [])
                    if evidence_against:
                        st.markdown("*Concerns:*")
                        for e in evidence_against[:2]:
                            st.markdown(f"⚠️ {e}")
            
            # Contributions
            contributions = exp.get("contributions", [])
            if contributions:
                st.markdown("### What They Could Contribute")
                for c in contributions[:3]:
                    st.info(f"**{c.get('contribution_type', 'Unknown')}:** {c.get('description', 'N/A')} → *Value: {c.get('value_to_project', 'N/A')}*")
            
            # Email template
            template = exp.get("email_template", {})
            if template:
                st.markdown("### 📧 Ready-to-Send Email Template")
                st.code(f"Subject: {template.get('subject', 'N/A')}\n\n{template.get('body', 'N/A')}", language="text")


def display_quality_checks(checks: Dict):
    """Display quality checks / self-assessment panel"""
    if not checks:
        return
    
    st.markdown("## ✅ Quality Checks (Self-Assessment)")
    
    cols = st.columns(4)
    
    with cols[0]:
        status = "✅" if checks.get("all_citations_verified") else "❌"
        st.markdown(f"{status} **All Citations Verified**")
        
        status = "✅" if checks.get("top3_papers_used") else "⚠️"
        st.markdown(f"{status} **Top 3 Papers Used**")
    
    with cols[1]:
        num_count = checks.get("numbers_count", 0)
        status = "✅" if num_count >= 15 else "⚠️" if num_count >= 10 else "❌"
        st.metric(f"{status} Numbers", num_count)
    
    with cols[2]:
        status = "✅" if checks.get("methodology_has_code") else "⚠️"
        st.markdown(f"{status} **Methodology Has Code**")
        
        status = "✅" if checks.get("experts_have_email") else "⚠️"
        st.markdown(f"{status} **Experts Have Email**")
    
    with cols[3]:
        compliance = checks.get("overall_compliance", 0)
        color = "normal" if compliance >= 70 else "off"
        st.metric("Compliance", f"{compliance}%", delta=None)
    
    # Vague words found
    vague = checks.get("vague_words_found", [])
    if vague:
        with st.expander("⚠️ Vague Words Found", expanded=False):
            for v in vague[:10]:
                st.warning(v)
    
    # Fabricated citations
    fabricated = checks.get("fabricated_citations", [])
    if fabricated:
        with st.expander("❌ Fabricated Citations", expanded=True):
            for f in fabricated[:5]:
                st.error(f)


def main():
    st.markdown('<h1 class="main-header">🔬 ScienceBridge</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Research Discovery with REAL Papers, Cross-Domain Search & Expert Discovery</p>', unsafe_allow_html=True)
    
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
        - 🌐 **Cross-Domain Search** (3+ related fields)
        - 👨‍🔬 **OpenAlex Authors** (50M+ researchers)
        - 📦 **HuggingFace** (datasets)
        - 💻 **GitHub** (implementation repos)
        """)
        
        st.markdown("---")
        st.caption("All data is fetched LIVE from APIs. No local storage.")
    
    # Main content
    st.markdown("### 1️⃣ Describe Your Research Question")
    
    query = st.text_area(
        "What problem are you trying to solve?",
        placeholder="Example: I'm researching quantum entanglement for secure communication. What cross-domain approaches from cryptography and signal processing could enhance QKD protocols?",
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
                (15, "📚 Searching primary field papers..."),
                (30, "🌐 Searching cross-domain fields..."),
                (45, "👨‍🔬 Finding expert researchers..."),
                (60, "📊 Finding relevant datasets..."),
                (75, "💻 Finding implementation repos..."),
                (90, "🤖 Generating ENHANCED hypothesis..."),
            ]
            
            for prog, msg in steps:
                status.text(msg)
                progress.progress(prog)
                time.sleep(0.3)
            
            result = call_api(query, field, creativity)
            
            progress.progress(100)
            status.text("✅ Complete!")
        
        if result.get("success"):
            st.success("🎉 Hypothesis generated with REAL research data + cross-domain connections!")
            
            # NEW: Display quality score FIRST
            quality = result.get("quality_score")
            if quality:
                display_quality_score(quality)
                st.markdown("---")
            
            # Display search stats
            stats = result.get("search_stats", {})
            display_search_stats(stats)
            
            # Show cross-domain and expert counts
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🌐 Cross-Domain Fields Searched", stats.get("cross_domain_fields_searched", 0))
            with col2:
                st.metric("👨‍🔬 Experts Found", stats.get("experts_found", 0))
            
            st.markdown("---")
            
            # Get hypothesis for new display
            hypothesis = result.get("hypothesis", {})
            
            # NEW: Display hypothesis in new format
            display_new_hypothesis(hypothesis)
            
            st.markdown("---")
            
            # NEW: Display problem context (SOTA, failed attempts, gap)
            if hypothesis.get("problem_context"):
                display_problem_context(hypothesis.get("problem_context"))
                st.markdown("---")
            
            # Display cross-domain discoveries
            # Check for new format first, then old format
            cross_conns = hypothesis.get("cross_domain_connections", [])
            if cross_conns:
                st.markdown("## 🔄 Cross-Domain Connections")
                for conn in cross_conns[:3]:
                    with st.expander(f"🔗 {conn.get('source_domain', 'Field A')} → {conn.get('target_domain', 'Field B')}", expanded=True):
                        st.markdown(f"**Technique:** {conn.get('source_technique', 'N/A')}")
                        st.markdown(f"**Source Paper:** {conn.get('source_paper', 'N/A')}")
                        st.markdown(f"**Finding:** {conn.get('source_finding', 'N/A')}")
                        st.markdown(f"**Transfer Mechanism:** {conn.get('transfer_mechanism', 'N/A')}")
                        st.markdown(f"**Why Non-Obvious:** {conn.get('why_nonobvious', 'N/A')}")
                st.markdown("---")
            else:
                display_cross_domain(result.get("cross_domain", {}))
                st.markdown("---")
            
            # NEW: Display methodology in new format
            methodology = hypothesis.get("methodology", [])
            if methodology:
                display_methodology_new(methodology)
                st.markdown("---")
            
            # NEW: Display comparison table
            comparison = hypothesis.get("comparison_table")
            if comparison:
                display_comparison_table(comparison)
                st.markdown("---")
            
            # NEW: Display validation metrics
            validation = hypothesis.get("validation_metrics")
            if validation:
                display_validation_metrics(validation)
                st.markdown("---")
            
            # NEW: Display risk assessment
            risks = hypothesis.get("risk_assessment", [])
            if risks:
                display_risk_assessment(risks)
                st.markdown("---")
            
            # ============== NEW SECTIONS (A-F) from updatesprompt.md ==============
            
            # NEW: Novelty Analysis
            novelty = hypothesis.get("novelty_analysis")
            if novelty:
                display_novelty_analysis(novelty)
                st.markdown("---")
            
            # NEW (A): Why This Hasn't Been Done Before
            why_not = hypothesis.get("why_not_done_before")
            if why_not:
                display_why_not_done_before(why_not)
                st.markdown("---")
            
            # NEW (B): Alternative Approaches Rejected
            alternatives = hypothesis.get("alternatives_rejected")
            if alternatives:
                display_alternatives_rejected(alternatives)
                st.markdown("---")
            
            # NEW (C): Preliminary Data / Proof of Concept
            prelim = hypothesis.get("preliminary_data")
            if prelim:
                display_preliminary_data(prelim)
                st.markdown("---")
            
            # NEW (D): Broader Impact
            impact = hypothesis.get("broader_impact")
            if impact:
                display_broader_impact(impact)
                st.markdown("---")
            
            # NEW (E): Funding Opportunities
            funding = hypothesis.get("funding_plan")
            if funding:
                display_funding_opportunities(funding)
                st.markdown("---")
            
            # NEW (F): IP Landscape
            ip = hypothesis.get("ip_landscape")
            if ip:
                display_ip_landscape(ip)
                st.markdown("---")
            
            # NEW: Enhanced experts with email templates (if available)
            enhanced_exp = hypothesis.get("enhanced_experts", [])
            if enhanced_exp:
                display_enhanced_experts(enhanced_exp)
                st.markdown("---")
            else:
                # Fall back to expert_collaborators or real_experts
                experts = hypothesis.get("expert_collaborators", [])
                if experts:
                    st.markdown("## 👨‍🔬 Recommended Expert Collaborators")
                    for exp in experts[:3]:
                        with st.expander(f"🔬 {exp.get('name', 'Unknown')} - {exp.get('institution', 'Unknown')}", expanded=False):
                            st.markdown(f"**Expertise:** {exp.get('expertise', 'N/A')}")
                            st.markdown(f"**h-index:** {exp.get('h_index', 'N/A')}")
                            st.markdown(f"**Why Contact:** {exp.get('why_contact', 'N/A')}")
                            st.markdown(f"**Collaboration Likelihood:** {exp.get('collaboration_likelihood', 'N/A')}")
                            papers = exp.get("relevant_papers", [])
                            if papers:
                                st.markdown("**Relevant Papers:**")
                                for p in papers[:2]:
                                    st.markdown(f"- {p}")
                    st.markdown("---")
                else:
                    display_real_experts(result.get("real_experts", []))
                    st.markdown("---")
            
            # NEW: Quality Checks (self-assessment)
            quality_checks = hypothesis.get("quality_checks")
            if quality_checks:
                display_quality_checks(quality_checks)
                st.markdown("---")
            
            # Display real papers
            display_real_papers(result.get("real_papers", []))
            
            st.markdown("---")
            
            # Display datasets
            # Check new format first
            rel_datasets = hypothesis.get("relevant_datasets", [])
            if rel_datasets:
                st.markdown("## 📊 Relevant Datasets")
                for d in rel_datasets[:5]:
                    with st.container():
                        st.markdown(f"**{d.get('name', 'Unknown')}** - {d.get('source', 'N/A')}")
                        st.markdown(f"*Relevance:* {d.get('relevance', 'N/A')}")
                        st.markdown(f"*Specific Use:* {d.get('specific_use', 'N/A')}")
                        st.markdown("---")
            else:
                display_real_datasets(result.get("real_datasets", []))
            
            st.markdown("---")
            
            # Display repos
            rel_repos = hypothesis.get("relevant_code", [])
            if rel_repos:
                st.markdown("## 💻 Relevant Code Repositories")
                for r in rel_repos[:5]:
                    with st.container():
                        st.markdown(f"**{r.get('repo_name', 'Unknown')}**")
                        st.markdown(f"*Relevance:* {r.get('relevance', 'N/A')}")
                        st.markdown(f"*Specific Use:* {r.get('specific_use', 'N/A')}")
                        st.markdown("---")
            else:
                display_real_repos(result.get("real_repos", []))
            
        else:
            st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
            st.info("Make sure the backend is running: `python -m uvicorn src.api.main:app --reload`")


if __name__ == "__main__":
    main()

