"""
Visualization and Reporting

Generate visualizations and reports for hypotheses and research analysis.
"""

from typing import Dict, List, Optional
from datetime import datetime
import json
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """Generates reports from hypothesis engine results"""
    
    def __init__(self):
        self.reports = []
    
    def generate_html_report(self, results: Dict) -> str:
        """Generate HTML report from results"""
        hypotheses = results.get('hypotheses', [])
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Hypothesis Generation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; }}
        .hypothesis {{ border: 1px solid #ddd; margin: 20px 0; padding: 15px; border-radius: 5px; }}
        .score {{ background: #3498db; color: white; padding: 5px 10px; border-radius: 3px; }}
        .high {{ background: #27ae60; }}
        .medium {{ background: #f39c12; }}
        .low {{ background: #e74c3c; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Hypothesis Generation Report</h1>
        <p>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</p>
        <p>Query: {results.get('query', 'N/A')}</p>
    </div>
    
    <h2>Summary</h2>
    <ul>
        <li>Total Hypotheses: {len(hypotheses)}</li>
        <li>Execution Time: {results.get('execution_time', 0):.2f}s</li>
    </ul>
    
    <h2>Generated Hypotheses</h2>
"""
        
        for i, hyp in enumerate(hypotheses, 1):
            novelty = hyp.get('novelty_score', 0)
            score_class = 'high' if novelty > 0.7 else 'medium' if novelty > 0.4 else 'low'
            
            html += f"""
    <div class="hypothesis">
        <h3>#{i}: {hyp.get('title', 'Untitled')}</h3>
        <p>{hyp.get('description', 'No description')}</p>
        <span class="score {score_class}">Novelty: {novelty:.2f}</span>
        <span class="score">Feasibility: {hyp.get('feasibility_score', 0):.2f}</span>
        <span class="score">Impact: {hyp.get('impact_potential', 0):.2f}</span>
    </div>
"""
        
        html += """
</body>
</html>
"""
        return html
    
    def generate_markdown_report(self, results: Dict) -> str:
        """Generate Markdown report"""
        lines = [
            f"# Hypothesis Generation Report",
            f"",
            f"**Generated**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Query**: {results.get('query', 'N/A')}",
            f"",
            f"## Summary",
            f"- Total Hypotheses: {len(results.get('hypotheses', []))}",
            f"- Execution Time: {results.get('execution_time', 0):.2f}s",
            f"",
            f"## Hypotheses",
            f""
        ]
        
        for i, hyp in enumerate(results.get('hypotheses', []), 1):
            lines.extend([
                f"### {i}. {hyp.get('title', 'Untitled')}",
                f"",
                f"{hyp.get('description', 'No description')}",
                f"",
                f"| Metric | Score |",
                f"|--------|-------|",
                f"| Novelty | {hyp.get('novelty_score', 0):.2f} |",
                f"| Feasibility | {hyp.get('feasibility_score', 0):.2f} |",
                f"| Impact | {hyp.get('impact_potential', 0):.2f} |",
                f""
            ])
        
        return '\n'.join(lines)
    
    def export_json(self, results: Dict, filepath: str):
        """Export results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Exported report to {filepath}")
    
    def generate_comparison_table(self, hypotheses: List[Dict]) -> str:
        """Generate comparison table for hypotheses"""
        lines = [
            "| Rank | Title | Novelty | Feasibility | Impact | Composite |",
            "|------|-------|---------|-------------|--------|-----------|"
        ]
        
        for i, hyp in enumerate(hypotheses, 1):
            composite = (hyp.get('novelty_score', 0) * 0.4 + 
                        hyp.get('feasibility_score', 0) * 0.3 + 
                        hyp.get('impact_potential', 0) * 0.3)
            
            lines.append(
                f"| {i} | {hyp.get('title', 'Untitled')[:40]}... | "
                f"{hyp.get('novelty_score', 0):.2f} | "
                f"{hyp.get('feasibility_score', 0):.2f} | "
                f"{hyp.get('impact_potential', 0):.2f} | "
                f"{composite:.2f} |"
            )
        
        return '\n'.join(lines)
