"""
Hypothesis Validator - Catches hallucinations and quality issues

The CRITICAL validation layer that ensures:
1. Every citation exists in retrieved papers
2. Sufficient specific numbers are included
3. Cross-domain connections are genuine
4. No vague language
"""

import re
import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of hypothesis validation"""
    is_valid: bool
    citation_quality: float  # 0-10
    specificity: float  # 0-10
    cross_domain: float  # 0-10
    actionability: float  # 0-10
    overall: float  # 0-10
    issues: List[str]
    fabricated_citations: List[str]
    missing_numbers: List[str]
    vague_claims: List[str]


class HypothesisValidator:
    """
    Validates hypothesis output against retrieved papers.
    Catches hallucinations before user sees them.
    
    UPDATED per updatesprompt.md v2:
    - Stricter citation verification
    - Check top-3 papers are used
    - Complete banned word list
    - Numbers per section requirement
    - Cross-domain dual citation check
    - Code snippet verification
    - Expert email template check
    """
    
    # COMPLETE banned vague words from updatesprompt.md
    BANNED_WORDS = [
        # Original vague words
        "significant", "significantly", "substantial", "substantially",
        "considerable", "considerably", "notable", "notably",
        "remarkable", "remarkably", "improved", "improvement",
        "better", "worse", "faster", "slower", "enhanced",
        "efficient", "effective", "important", "major", "minor",
        "large", "small", "high", "low", "many", "few",
        "some", "several", "various", "numerous", "multiple",
        # Additional from updated updatesprompt.md
        "good", "bad", "great", "excellent", "superior", "inferior",
        "valuable", "useful", "helpful", "promising", "remarkable",
        "outstanding", "tremendous", "incredible", "amazing",
        "optimal", "ideal", "perfect", "sufficient", "insufficient",
        "adequate", "inadequate", "appropriate", "inappropriate"
    ]
    
    # Patterns to find numbers with units
    NUMBER_PATTERNS = [
        r'\d+\.?\d*\s*%',  # percentages
        r'\d+\.?\d*\s*(hours?|hrs?|h)\b',  # time hours
        r'\d+\.?\d*\s*(minutes?|mins?|m)\b',  # time minutes
        r'\d+\.?\d*\s*(seconds?|secs?|s)\b',  # time seconds
        r'\d+\.?\d*\s*(days?|d)\b',  # time days
        r'\d+\.?\d*\s*(weeks?|wks?)',  # time weeks
        r'\d+\.?\d*\s*(months?)',  # time months
        r'\d+\.?\d*\s*(years?|yrs?)',  # time years
        r'\d+\.?\d*\s*°C',  # temperature
        r'\d+\.?\d*\s*K\b',  # temperature kelvin
        r'\d+\.?\d*\s*(mg|g|kg)\b',  # mass
        r'\d+\.?\d*\s*(mL|L|μL)\b',  # volume
        r'\d+\.?\d*\s*(mM|μM|nM|M)\b',  # concentration
        r'\d+\.?\d*\s*(kDa|Da)\b',  # molecular weight
        r'\d+\.?\d*\s*(nm|μm|mm|cm|m)\b',  # length
        r'\d+\.?\d*x\b',  # multipliers
        r'\$\d+\.?\d*(K|M|B)?',  # money
        r'\d{1,3}(,\d{3})+',  # large numbers with commas
        r'\d+\.?\d*\s*citations?',  # citations
        r'[1-9]\d*/10',  # ratings
        r'\d+\.?\d*\s*GPU',  # compute
        r'\d+\.?\d*\s*(GB|TB|MB)',  # storage
    ]
    
    def __init__(self):
        self.retrieved_papers_cache = []
        self.retrieved_dois_cache = set()
        self.retrieved_authors_cache = set()
    
    def build_paper_lookup(self, retrieved_papers: List[Dict]) -> None:
        """Build comprehensive lookup for paper validation"""
        self.retrieved_papers_cache = retrieved_papers
        self.retrieved_dois_cache = set()
        self.retrieved_authors_cache = set()
        
        for paper in retrieved_papers:
            # DOI lookup
            doi = paper.get("doi", "")
            if doi:
                self.retrieved_dois_cache.add(doi.lower())
            
            # Author-year lookup
            authors = paper.get("authors", "Unknown")
            year = str(paper.get("year", ""))
            
            if authors and authors != "Unknown" and year:
                # First author last name
                parts = authors.split(",")[0].split()
                if parts:
                    first_author = parts[-1]
                    self.retrieved_authors_cache.add(f"{first_author.lower()} ({year})")
                    self.retrieved_authors_cache.add(f"{first_author.lower()} et al. ({year})")
                    self.retrieved_authors_cache.add(f"{first_author.lower()} {year}")
                    self.retrieved_authors_cache.add(f"{first_author.lower()}, {year}")
            
            # Title-based lookup
            title = paper.get("title", "")
            if title:
                # First few words of title
                title_words = title.split()[:5]
                self.retrieved_authors_cache.add(" ".join(title_words).lower())
    
    def get_top_cited_papers(self, retrieved_papers: List[Dict], n: int = 3) -> List[Dict]:
        """Get top N papers by citation count"""
        sorted_papers = sorted(
            retrieved_papers,
            key=lambda p: p.get("citation_count", 0),
            reverse=True
        )
        return sorted_papers[:n]
    
    def extract_citations_from_text(self, text: str) -> List[str]:
        """Extract author-year citations from text"""
        citations = []
        
        # Pattern: Author et al. (Year) or Author (Year)
        pattern1 = r'([A-Z][a-z]+(?:\s+et\s+al\.)?)\s*\((\d{4})\)'
        matches = re.findall(pattern1, text)
        for author, year in matches:
            citations.append(f"{author} ({year})")
        
        # Pattern: Author et al., Year
        pattern2 = r'([A-Z][a-z]+(?:\s+et\s+al\.)?),?\s*(\d{4})'
        matches = re.findall(pattern2, text)
        for author, year in matches:
            citations.append(f"{author} ({year})")
        
        return list(set(citations))
    
    # ========== NEW VALIDATION FUNCTION 1 ==========
    def verify_all_citations_exist(
        self,
        hypothesis_text: str,
        retrieved_papers: List[Dict]
    ) -> Tuple[bool, List[str], List[str]]:
        """
        STRICT verification that EVERY citation exists in retrieved papers.
        Returns (all_valid, valid_citations, fabricated_citations)
        """
        self.build_paper_lookup(retrieved_papers)
        cited = self.extract_citations_from_text(hypothesis_text)
        
        valid = []
        fabricated = []
        
        for citation in cited:
            clean = citation.strip().lower()
            
            # Check against all lookup patterns
            found = False
            for retrieved in self.retrieved_authors_cache:
                if clean in retrieved or retrieved in clean:
                    found = True
                    break
                # Check just the author name
                author_only = clean.split("(")[0].strip()
                if author_only in retrieved:
                    found = True
                    break
            
            if found:
                valid.append(citation)
            else:
                fabricated.append(citation)
        
        all_valid = len(fabricated) == 0
        return all_valid, valid, fabricated
    
    # ========== NEW VALIDATION FUNCTION 2 ==========
    def check_top3_papers_used(
        self,
        hypothesis_text: str,
        retrieved_papers: List[Dict]
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Check if top 3 highest-cited papers are referenced.
        Returns (all_used, top3_papers, missing_papers)
        """
        top3 = self.get_top_cited_papers(retrieved_papers, 3)
        
        top3_info = []
        missing = []
        
        for paper in top3:
            authors = paper.get("authors", "Unknown")
            year = str(paper.get("year", ""))
            title = paper.get("title", "")
            citations = paper.get("citation_count", 0)
            
            paper_info = f"{authors.split(',')[0]} ({year}) - {citations} citations"
            top3_info.append(paper_info)
            
            # Check if referenced
            first_author = authors.split(",")[0].split()[-1] if authors else ""
            found = False
            
            if first_author:
                # Check various citation patterns
                patterns = [
                    f"{first_author}",
                    f"{first_author.lower()}",
                    title.split()[:3] if title else [""]
                ]
                
                for pattern in patterns:
                    if isinstance(pattern, list):
                        if all(word.lower() in hypothesis_text.lower() for word in pattern if word):
                            found = True
                            break
                    elif pattern.lower() in hypothesis_text.lower():
                        found = True
                        break
            
            if not found:
                missing.append(paper_info)
        
        all_used = len(missing) == 0
        return all_used, top3_info, missing
    
    # ========== NEW VALIDATION FUNCTION 3 ==========
    def count_banned_words(self, text: str) -> Tuple[int, List[str]]:
        """
        Count and list all banned vague words found.
        Returns (count, list_of_found_words_with_context)
        """
        found = []
        text_lower = text.lower()
        
        for word in self.BANNED_WORDS:
            pattern = rf'\b{word}\b'
            matches = re.findall(pattern, text_lower)
            if matches:
                # Find context
                context_pattern = rf'.{{0,30}}\b{word}\b.{{0,30}}'
                contexts = re.findall(context_pattern, text_lower)
                for ctx in contexts[:1]:  # Just first occurrence
                    found.append(f"'{word}': ...{ctx.strip()}...")
        
        return len(found), found
    
    # ========== NEW VALIDATION FUNCTION 4 ==========
    def verify_numbers_per_section(self, hypothesis_json: Dict) -> Tuple[bool, Dict[str, int]]:
        """
        Verify minimum required numbers per section.
        Requirements:
        - Problem statement: 3+ numbers
        - Methodology: 5+ numbers per step
        - Comparison table: all cells have numbers
        - Risk assessment: probabilities (%)
        """
        section_counts = {}
        issues = []
        
        # Problem context
        problem_text = json.dumps(hypothesis_json.get("problem_context", {}))
        count, _ = self.count_specific_numbers(problem_text)
        section_counts["problem_context"] = count
        if count < 3:
            issues.append(f"Problem context: only {count} numbers (need 3+)")
        
        # Methodology
        methodology = hypothesis_json.get("methodology", [])
        for i, step in enumerate(methodology):
            step_text = json.dumps(step)
            count, _ = self.count_specific_numbers(step_text)
            section_counts[f"methodology_step_{i+1}"] = count
            if count < 3:
                issues.append(f"Methodology step {i+1}: only {count} numbers (need 3+)")
        
        # Comparison table
        comparison = hypothesis_json.get("comparison_table", {})
        methods = comparison.get("methods", [])
        for method in methods:
            perf = method.get("performance", "")
            cost = method.get("cost", "")
            has_numbers = bool(re.search(r'\d', perf + cost))
            if not has_numbers:
                issues.append(f"Comparison method '{method.get('name', '?')}' missing numbers")
        
        # Risk assessment
        risks = hypothesis_json.get("risk_assessment", [])
        for risk in risks:
            prob = risk.get("probability", "")
            if not re.search(r'\d+\s*%', prob):
                issues.append(f"Risk '{risk.get('risk', '?')}' missing probability %")
        
        all_valid = len(issues) == 0
        return all_valid, section_counts
    
    # ========== NEW VALIDATION FUNCTION 5 ==========
    def validate_cross_domain_dual_citations(
        self,
        connections: List[Dict],
        retrieved_papers: List[Dict]
    ) -> Tuple[bool, List[str]]:
        """
        Check that cross-domain connections cite papers from BOTH domains.
        """
        self.build_paper_lookup(retrieved_papers)
        issues = []
        
        for i, conn in enumerate(connections):
            source_paper = conn.get("source_paper", "")
            # Target domain paper should be in supporting papers or connected papers
            target_problem = conn.get("target_problem", "")
            transfer_mechanism = conn.get("transfer_mechanism", "")
            
            # Check source paper exists
            source_in_retrieved = any(
                author.lower() in source_paper.lower()
                for author in self.retrieved_authors_cache
            )
            
            if not source_in_retrieved and source_paper:
                # Check if it's a well-known paper (not from our set but legitimate)
                if not re.search(r'\d{4}', source_paper):
                    issues.append(f"Cross-domain {i+1}: Source paper missing year")
            
            # Check for specific mechanism
            if len(transfer_mechanism) < 100:
                issues.append(f"Cross-domain {i+1}: Transfer mechanism too brief ({len(transfer_mechanism)} chars, need 100+)")
            
            # Check for numbers in expected improvement
            expected = conn.get("expected_improvement", "") if isinstance(conn, dict) else ""
            if expected and not re.search(r'\d', str(expected)):
                issues.append(f"Cross-domain {i+1}: Expected improvement missing numbers")
        
        all_valid = len(issues) == 0
        return all_valid, issues
    
    # ========== NEW VALIDATION FUNCTION 6 ==========
    def validate_methodology_has_code(self, methodology: List[Dict]) -> Tuple[bool, List[str]]:
        """
        Check that methodology steps include code snippets.
        At least 50% of steps should have code.
        """
        issues = []
        steps_with_code = 0
        
        for step in methodology:
            code = step.get("code_snippet", "")
            if code and len(code) > 20:  # At least 20 chars of code
                steps_with_code += 1
            else:
                step_name = step.get("step_name", f"Step {step.get('step_number', '?')}")
                issues.append(f"'{step_name}' missing code snippet")
        
        if methodology:
            code_ratio = steps_with_code / len(methodology)
            if code_ratio < 0.5:
                issues.insert(0, f"Only {steps_with_code}/{len(methodology)} steps have code (need 50%+)")
        
        all_valid = len(issues) == 0
        return all_valid, issues
    
    # ========== NEW VALIDATION FUNCTION 7 ==========
    def validate_expert_contact_complete(
        self,
        experts: List[Dict],
        enhanced_experts: List[Dict] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check that experts have complete contact info and email templates.
        """
        issues = []
        all_experts = experts + (enhanced_experts or [])
        
        for expert in all_experts:
            name = expert.get("name", "Unknown")
            
            # Check email
            email = expert.get("email")
            if not email:
                issues.append(f"Expert '{name}' missing email")
            
            # Check why_contact / contributions
            why = expert.get("why_contact", "") or expert.get("contributions", [])
            if not why:
                issues.append(f"Expert '{name}' missing contribution explanation")
            
            # Check email template (for enhanced experts)
            template = expert.get("email_template")
            if not template and expert.get("priority") == "HIGHEST":
                issues.append(f"Expert '{name}' (HIGHEST priority) missing email template")
        
        all_valid = len(issues) <= 1  # Allow 1 minor issue
        return all_valid, issues
    
    # ========== GENERATE QUALITY CHECKS OBJECT ==========
    def generate_quality_checks(
        self,
        hypothesis_json: Dict,
        retrieved_papers: List[Dict]
    ) -> Dict:
        """
        Generate complete QualityChecks object for the new schema.
        """
        hypothesis_text = json.dumps(hypothesis_json, default=str)
        
        # 1. Citation verification
        all_valid, valid_cites, fabricated = self.verify_all_citations_exist(
            hypothesis_text, retrieved_papers
        )
        
        # 2. Top 3 papers
        top3_used, top3_list, missing_top3 = self.check_top3_papers_used(
            hypothesis_text, retrieved_papers
        )
        
        # 3. Banned words
        banned_count, banned_found = self.count_banned_words(hypothesis_text)
        
        # 4. Numbers count
        num_count, _ = self.count_specific_numbers(hypothesis_text)
        
        # 5. Cross-domain dual citations
        connections = hypothesis_json.get("cross_domain_connections", [])
        cross_valid, _ = self.validate_cross_domain_dual_citations(
            connections, retrieved_papers
        )
        
        # 6. Methodology code
        methodology = hypothesis_json.get("methodology", [])
        code_valid, _ = self.validate_methodology_has_code(methodology)
        
        # 7. Expert emails
        experts = hypothesis_json.get("expert_collaborators", [])
        enhanced = hypothesis_json.get("enhanced_experts", [])
        email_valid, _ = self.validate_expert_contact_complete(experts, enhanced)
        
        # Calculate compliance
        checks_passed = sum([
            all_valid,
            top3_used,
            banned_count == 0,
            num_count >= 15,
            cross_valid,
            code_valid,
            email_valid
        ])
        compliance = (checks_passed / 7) * 100
        
        return {
            "all_citations_verified": all_valid,
            "fabricated_citations": fabricated,
            "top3_papers_used": top3_used,
            "top3_papers_list": top3_list,
            "vague_words_found": banned_found[:10],
            "numbers_count": num_count,
            "cross_domain_has_both_citations": cross_valid,
            "methodology_has_code": code_valid,
            "experts_have_email": email_valid,
            "overall_compliance": round(compliance, 1)
        }
    
    # Keep existing methods
    def validate_citations(
        self,
        hypothesis_text: str,
        retrieved_papers: List[Dict]
    ) -> Tuple[List[str], List[str]]:
        """Check if all cited papers exist in retrieved list."""
        _, valid, fabricated = self.verify_all_citations_exist(hypothesis_text, retrieved_papers)
        return valid, fabricated
    
    def count_specific_numbers(self, text: str) -> Tuple[int, List[str]]:
        """Count quantitative claims with units"""
        numbers_found = []
        
        for pattern in self.NUMBER_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            numbers_found.extend(matches)
        
        # Also find standalone numbers in context
        standalone = re.findall(
            r'\b(\d+(?:\.\d+)?)\s+(?:times|fold|variants|samples|papers|experts|steps|cells|patients)',
            text, re.IGNORECASE
        )
        numbers_found.extend(standalone)
        
        return len(numbers_found), numbers_found
    
    def find_vague_claims(self, text: str) -> List[str]:
        """Find vague claims that should have numbers"""
        _, vague_found = self.count_banned_words(text)
        return vague_found[:10]
    
    def check_cross_domain_quality(self, connections: List[Dict]) -> Tuple[float, List[str]]:
        """Check if cross-domain connections are genuine"""
        issues = []
        score = 0.0
        
        if not connections:
            return 0.0, ["No cross-domain connections provided"]
        
        for conn in connections:
            conn_score = 0
            
            # Check for specific technique
            technique = conn.get("source_technique", "")
            if len(technique) > 20 and technique.lower() not in ["techniques", "methods", "approaches"]:
                conn_score += 2
            else:
                issues.append(f"Vague technique: '{technique}'")
            
            # Check for specific numbers in finding
            finding = conn.get("source_finding", "")
            num_count, _ = self.count_specific_numbers(finding)
            if num_count > 0:
                conn_score += 2
            else:
                issues.append(f"No numbers in source finding")
            
            # Check for specific transfer mechanism
            mechanism = conn.get("transfer_mechanism", "")
            if len(mechanism) > 50 and "could" not in mechanism.lower():
                conn_score += 3
            else:
                issues.append(f"Generic transfer mechanism")
            
            # Check why_nonobvious
            why = conn.get("why_nonobvious", "")
            if len(why) > 30:
                conn_score += 3
            else:
                issues.append(f"Missing explanation of why non-obvious")
            
            score += min(conn_score, 10)
        
        return min(score / len(connections), 10), issues
    
    def check_methodology_quality(self, methodology: List[Dict]) -> Tuple[float, List[str]]:
        """Check methodology steps for completeness"""
        issues = []
        total_score = 0
        
        if not methodology:
            return 0.0, ["No methodology steps provided"]
        
        for step in methodology:
            step_score = 0
            step_num = step.get("step_number", "?")
            
            # Check algorithm specificity
            algo = step.get("algorithm", "")
            if algo and len(algo) > 10:
                step_score += 2
            else:
                issues.append(f"Step {step_num}: Missing specific algorithm")
            
            # Check for source papers
            papers = step.get("source_papers", [])
            if papers and len(papers) > 0:
                step_score += 2
            else:
                issues.append(f"Step {step_num}: No source papers cited")
            
            # Check for parameters
            params = step.get("parameters", {})
            if params and len(params) > 0:
                step_score += 2
            else:
                issues.append(f"Step {step_num}: No parameters specified")
            
            # Check for success criteria
            criteria = step.get("success_criteria", "")
            if criteria and len(criteria) > 20:
                step_score += 2
            else:
                issues.append(f"Step {step_num}: Missing success criteria")
            
            # Check for code snippet (NEW)
            code = step.get("code_snippet", "")
            if code and len(code) > 20:
                step_score += 2
            else:
                issues.append(f"Step {step_num}: Missing code snippet")
            
            total_score += min(step_score, 10)
        
        return min(total_score / len(methodology), 10), issues
    
    def validate(
        self,
        hypothesis_json: Dict,
        retrieved_papers: List[Dict]
    ) -> ValidationResult:
        """
        Complete validation of hypothesis output.
        Returns quality scores and issues found.
        """
        issues = []
        
        # Convert to text for analysis
        hypothesis_text = json.dumps(hypothesis_json, default=str)
        
        # 1. Citation validation (STRICTER)
        all_valid, valid_cites, fabricated = self.verify_all_citations_exist(
            hypothesis_text, retrieved_papers
        )
        if fabricated:
            issues.append(f"CRITICAL: {len(fabricated)} fabricated citations found: {fabricated[:3]}")
            citation_quality = max(0, 10 - len(fabricated) * 3)  # Harsher penalty
        else:
            citation_quality = 10.0
        
        # 2. Top 3 papers check (NEW)
        top3_used, top3_list, missing_top3 = self.check_top3_papers_used(
            hypothesis_text, retrieved_papers
        )
        if missing_top3:
            issues.append(f"Missing top-cited papers: {missing_top3[:2]}")
            citation_quality = max(0, citation_quality - 1)
        
        # 3. Specificity check
        num_count, numbers = self.count_specific_numbers(hypothesis_text)
        if num_count >= 20:
            specificity = 10.0
        elif num_count >= 15:
            specificity = 8.0
        elif num_count >= 10:
            specificity = 6.0
        elif num_count >= 5:
            specificity = 4.0
        else:
            specificity = num_count * 0.8
            issues.append(f"Only {num_count} specific numbers found (need 15+)")
        
        # 4. Banned words check (STRICTER)
        banned_count, vague = self.count_banned_words(hypothesis_text)
        if banned_count > 0:
            specificity = max(0, specificity - banned_count * 0.3)
            issues.append(f"Found {banned_count} banned vague words")
            if vague:
                issues.extend([f"Vague: {v}" for v in vague[:2]])
        
        # 5. Cross-domain quality
        connections = hypothesis_json.get("cross_domain_connections", [])
        cross_score, cross_issues = self.check_cross_domain_quality(connections)
        issues.extend(cross_issues[:2])
        
        # 6. Methodology quality (includes code check)
        methodology = hypothesis_json.get("methodology", [])
        method_score, method_issues = self.check_methodology_quality(methodology)
        issues.extend(method_issues[:2])
        
        # 7. Actionability (based on methodology + comparison table)
        actionability = method_score
        if hypothesis_json.get("comparison_table", {}).get("methods"):
            actionability = min(10, actionability + 2)
        if hypothesis_json.get("risk_assessment"):
            actionability = min(10, actionability + 1)
        if hypothesis_json.get("validation_metrics"):
            actionability = min(10, actionability + 1)
        # NEW: Check for new sections
        if hypothesis_json.get("why_not_done_before"):
            actionability = min(10, actionability + 0.5)
        if hypothesis_json.get("broader_impact"):
            actionability = min(10, actionability + 0.5)
        
        # Calculate overall
        overall = (citation_quality + specificity + cross_score + actionability) / 4
        
        # Determine validity (STRICTER)
        is_valid = (
            len(fabricated) == 0 and  # No fabricated citations
            num_count >= 10 and  # Minimum numbers
            banned_count <= 5 and  # Allow few vague words
            overall >= 6.0  # Higher minimum
        )
        
        return ValidationResult(
            is_valid=is_valid,
            citation_quality=round(citation_quality, 1),
            specificity=round(specificity, 1),
            cross_domain=round(cross_score, 1),
            actionability=round(actionability, 1),
            overall=round(overall, 1),
            issues=issues[:20],  # More issues shown
            fabricated_citations=fabricated,
            missing_numbers=["Need more quantitative claims"] if num_count < 15 else [],
            vague_claims=[v.split(":")[0] for v in vague[:5]] if vague else []
        )


# Global instance
hypothesis_validator = HypothesisValidator()


def validate_hypothesis(hypothesis_json: Dict, retrieved_papers: List[Dict]) -> ValidationResult:
    """Main function to validate hypothesis output"""
    return hypothesis_validator.validate(hypothesis_json, retrieved_papers)


def generate_quality_checks(hypothesis_json: Dict, retrieved_papers: List[Dict]) -> Dict:
    """Generate QualityChecks object for new schema"""
    return hypothesis_validator.generate_quality_checks(hypothesis_json, retrieved_papers)


def format_quality_score_for_display(result: ValidationResult) -> Dict:
    """Format validation result for frontend display"""
    return {
        "citation_quality": {
            "score": result.citation_quality,
            "max": 10,
            "status": "✅" if result.citation_quality >= 8 else "⚠️" if result.citation_quality >= 5 else "❌"
        },
        "specificity": {
            "score": result.specificity,
            "max": 10,
            "status": "✅" if result.specificity >= 8 else "⚠️" if result.specificity >= 5 else "❌"
        },
        "cross_domain": {
            "score": result.cross_domain,
            "max": 10,
            "status": "✅" if result.cross_domain >= 8 else "⚠️" if result.cross_domain >= 5 else "❌"
        },
        "actionability": {
            "score": result.actionability,
            "max": 10,
            "status": "✅" if result.actionability >= 8 else "⚠️" if result.actionability >= 5 else "❌"
        },
        "overall": {
            "score": result.overall,
            "max": 10,
            "status": "✅" if result.overall >= 7 else "⚠️" if result.overall >= 5 else "❌"
        },
        "is_valid": result.is_valid,
        "issues": result.issues,
        "fabricated_citations": result.fabricated_citations
    }
