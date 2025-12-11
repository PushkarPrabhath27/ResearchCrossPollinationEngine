"""
Cross-Domain Research Engine - CORE INNOVATION
Searches MULTIPLE related fields to find analogous problems and solutions.

This is the KEY VALUE of ScienceBridge - finding unexpected connections.

FREE APIs Used:
- OpenAlex (250M+ papers, FREE, no key)
- arXiv (2M+ papers, FREE, no key)
"""

import requests
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CrossDomainConnection:
    """Represents a connection between fields"""
    source_field: str
    target_field: str
    source_concept: str
    target_concept: str
    analogy_explanation: str
    supporting_papers: List[Dict[str, Any]]
    connection_strength: float  # 0-1


class CrossDomainEngine:
    """
    Generates cross-domain queries and finds analogous problems.
    
    Example:
    Query: "quantum entanglement for secure communication"
    Primary Field: Physics
    
    Cross-Domain Searches:
    1. Computer Science: "error correction secure key exchange"
    2. Engineering: "signal transmission noisy channel"
    3. Biology: "reliable signal transmission noisy environment"
    
    Connections Found:
    - Reed-Solomon error correction (CS) → Quantum error correction
    - MIMO wireless (Engineering) → Multi-qubit transmission
    """
    
    # Field relationship mappings - which fields have analogous problems
    FIELD_CONNECTIONS = {
        "physics": {
            "related_fields": ["computer_science", "mathematics", "engineering", "chemistry"],
            "analogies": {
                "quantum": ["cryptography", "error correction", "information theory"],
                "wave": ["signal processing", "communications", "acoustics"],
                "particle": ["network flow", "graph theory", "optimization"],
                "energy": ["resource allocation", "power systems", "thermodynamics"],
                "entanglement": ["correlation", "distributed systems", "synchronization"]
            }
        },
        "biology": {
            "related_fields": ["chemistry", "medicine", "computer_science", "engineering"],
            "analogies": {
                "neural": ["machine learning", "network architecture", "signal processing"],
                "genetic": ["information theory", "error correction", "optimization"],
                "protein": ["molecular dynamics", "graph theory", "optimization"],
                "evolution": ["optimization", "genetic algorithms", "game theory"],
                "cell": ["distributed systems", "communication networks", "control systems"],
                "enzyme": ["catalysis", "reaction mechanisms", "industrial chemistry"],
                "degradation": ["polymer chemistry", "waste treatment", "recycling"]
            }
        },
        "computer_science": {
            "related_fields": ["mathematics", "physics", "engineering", "biology"],
            "analogies": {
                "algorithm": ["optimization", "control theory", "evolutionary biology"],
                "network": ["graph theory", "social systems", "neural networks"],
                "security": ["cryptography", "game theory", "signal processing"],
                "learning": ["statistics", "optimization", "neuroscience"],
                "distributed": ["physics", "social dynamics", "biology"]
            }
        },
        "chemistry": {
            "related_fields": ["physics", "biology", "materials_science", "medicine", "engineering"],
            "analogies": {
                "reaction": ["dynamics", "control systems", "network flow"],
                "molecular": ["graph theory", "optimization", "physics"],
                "catalysis": ["optimization", "control theory", "economics"],
                "synthesis": ["planning", "optimization", "manufacturing"],
                "degradation": ["enzyme engineering", "bioremediation", "polymer science"],
                "plastic": ["polymer chemistry", "waste management", "sustainability"]
            }
        },
        "medicine": {
            "related_fields": ["biology", "chemistry", "computer_science", "engineering"],
            "analogies": {
                "diagnosis": ["pattern recognition", "machine learning", "signal processing"],
                "treatment": ["optimization", "control theory", "game theory"],
                "drug": ["chemistry", "optimization", "graph theory"],
                "imaging": ["signal processing", "physics", "computer vision"]
            }
        },
        "engineering": {
            "related_fields": ["physics", "mathematics", "computer_science", "materials_science"],
            "analogies": {
                "control": ["optimization", "dynamics", "biology"],
                "signal": ["information theory", "physics", "neuroscience"],
                "design": ["optimization", "machine learning", "evolution"],
                "materials": ["physics", "chemistry", "biology"]
            }
        },
        "mathematics": {
            "related_fields": ["physics", "computer_science", "engineering", "economics"],
            "analogies": {
                "optimization": ["physics", "machine learning", "economics"],
                "statistics": ["machine learning", "physics", "biology"],
                "graph": ["networks", "chemistry", "social science"],
                "probability": ["physics", "machine learning", "economics"]
            }
        }
    }
    
    # PHASE 5: TECHNIQUE-BASED CROSS-DOMAIN MAPPING
    # Specific techniques that transfer between fields with HOW they transfer
    CROSS_DOMAIN_TECHNIQUES = {
        "biology": {
            "chemistry": [
                "oxidation catalyst mechanism for PET hydrolysis",
                "enzyme kinetics for reaction optimization",
                "biocatalyst stability under industrial conditions"
            ],
            "engineering": [
                "bioreactor design for enzyme production",
                "scale-up from lab to industrial",
                "continuous flow reactor optimization"
            ],
            "medicine": [
                "drug metabolism pathway design",
                "prodrug hydrolysis mechanisms",
                "biomarker detection assays"
            ],
            "materials_science": [
                "polymer degradation mechanisms",
                "chain scission kinetics",
                "crystallinity effects on degradation"
            ]
        },
        "chemistry": {
            "biology": [
                "directed evolution for enzyme optimization",
                "protein engineering for stability",
                "metabolic pathway engineering"
            ],
            "physics": [
                "molecular dynamics simulation",
                "quantum chemistry for reaction mechanisms",
                "spectroscopy for structure analysis"
            ],
            "engineering": [
                "chemical reactor optimization",
                "process intensification",
                "separation technology"
            ]
        },
        "physics": {
            "computer_science": [
                "quantum computing algorithms",
                "error correction codes",
                "cryptographic protocols"
            ],
            "engineering": [
                "signal processing for quantum channels",
                "noise reduction techniques",
                "sensor optimization"
            ],
            "mathematics": [
                "quantum information theory",
                "statistical mechanics",
                "optimization theory"
            ]
        },
        "computer_science": {
            "biology": [
                "machine learning for protein structure",
                "sequence analysis algorithms",
                "evolutionary algorithms"
            ],
            "chemistry": [
                "molecular property prediction",
                "reaction outcome prediction",
                "retrosynthesis planning"
            ],
            "physics": [
                "quantum algorithm design",
                "simulation optimization",
                "numerical methods"
            ]
        }
    }
    
    # Transfer mechanism templates for generating explanations
    TRANSFER_MECHANISMS = {
        "optimization": "Apply {source_technique} optimization approach from {source_field} to {target_problem} by adapting the objective function",
        "algorithm": "Adapt {source_technique} algorithm structure from {source_field} for {target_problem} with domain-specific constraints",
        "mechanism": "Transfer {source_technique} mechanistic understanding from {source_field} to explain {target_problem} processes",
        "method": "Apply {source_technique} experimental methods from {source_field} to characterize {target_problem}",
        "model": "Use {source_technique} predictive model from {source_field} to design solutions for {target_problem}"
    }
    
    # OpenAlex concept IDs for filtering
    FIELD_CONCEPT_IDS = {
        "physics": "C121332964",
        "biology": "C86803240",
        "computer_science": "C41008148",
        "chemistry": "C185592680",
        "medicine": "C71924100",
        "engineering": "C127413603",
        "mathematics": "C33923547",
        "materials_science": "C192562407",
        "economics": "C162324750"
    }
    
    OPENALEX_BASE = "https://api.openalex.org"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ScienceBridge/1.0 (research-cross-pollination; mailto:contact@sciencebridge.ai)"
        })
    
    def extract_key_concepts(self, query: str) -> List[str]:
        """Extract important technical concepts from query"""
        # Remove stop words
        stop_words = {"i'm", "i", "am", "are", "is", "the", "a", "an", "and", "or", "for", 
                      "to", "from", "that", "which", "with", "could", "would", "should",
                      "there", "their", "what", "how", "can", "help", "enhance", "improve",
                      "researching", "studying", "looking", "finding", "about", "using",
                      "want", "need", "like", "think", "believe", "see", "know"}
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        keywords = [w for w in words if w not in stop_words]
        
        # Prioritize technical terms
        technical_priority = ["quantum", "neural", "genetic", "molecular", "algorithm",
                              "cryptography", "entanglement", "protein", "learning",
                              "optimization", "signal", "communication", "network",
                              "security", "error", "correction", "distributed"]
        
        prioritized = [k for k in keywords if k in technical_priority]
        other = [k for k in keywords if k not in prioritized]
        
        return prioritized + other[:5]
    
    def generate_cross_domain_queries(self, query: str, primary_field: str) -> List[Dict[str, Any]]:
        """
        Generate queries for related fields based on analogous problems.
        
        Returns list of:
        {
            "field": "computer_science",
            "query": "error correction secure key exchange", 
            "reasoning": "BB84 needs error correction, classical crypto has 40 years of research"
        }
        """
        cross_queries = []
        concepts = self.extract_key_concepts(query)
        
        field_lower = primary_field.lower().replace(" ", "_")
        field_info = self.FIELD_CONNECTIONS.get(field_lower, {})
        related_fields = field_info.get("related_fields", ["computer_science", "engineering"])
        analogies = field_info.get("analogies", {})
        
        for related_field in related_fields[:3]:  # Top 3 related fields
            # Find analogous concepts
            cross_concepts = []
            reasoning_parts = []
            
            for concept in concepts[:3]:
                if concept in analogies:
                    cross_concepts.extend(analogies[concept][:2])
                    reasoning_parts.append(f"{concept} in {primary_field} relates to {', '.join(analogies[concept][:2])} in {related_field}")
            
            if not cross_concepts:
                # Default cross-domain terms
                cross_concepts = concepts[:3]
            
            # Build cross-domain query
            cross_query = " ".join(cross_concepts[:4])
            reasoning = "; ".join(reasoning_parts) if reasoning_parts else f"Searching {related_field} for analogous approaches to {concepts[0] if concepts else 'this problem'}"
            
            cross_queries.append({
                "field": related_field,
                "query": cross_query,
                "reasoning": reasoning,
                "original_concepts": concepts[:3]
            })
        
        logger.info(f"[CrossDomain] Generated {len(cross_queries)} cross-domain queries")
        return cross_queries
    
    def search_field(self, query: str, field: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search OpenAlex for papers in a specific field"""
        papers = []
        
        try:
            concept_id = self.FIELD_CONCEPT_IDS.get(field.lower())
            filter_param = f"&filter=concepts.id:{concept_id}" if concept_id else ""
            
            url = f"{self.OPENALEX_BASE}/works?search={query}&per-page={max_results}&sort=relevance_score:desc{filter_param}"
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            for work in data.get("results", []):
                # Extract author info
                authorships = work.get("authorships", [])
                authors = ", ".join([
                    a.get("author", {}).get("display_name", "Unknown")
                    for a in authorships[:3]
                ])
                if len(authorships) > 3:
                    authors += " et al."
                
                # Get first author ID for expert fetching
                first_author_id = None
                if authorships:
                    first_author_id = authorships[0].get("author", {}).get("id", "")
                
                # Get DOI
                doi = work.get("doi", "")
                if doi:
                    doi = doi.replace("https://doi.org/", "")
                
                # Get abstract
                abstract = ""
                inverted_abstract = work.get("abstract_inverted_index", {})
                if inverted_abstract:
                    positions = []
                    for word, pos_list in inverted_abstract.items():
                        for pos in pos_list:
                            positions.append((pos, word))
                    positions.sort()
                    abstract = " ".join([word for _, word in positions])[:500]
                
                paper = {
                    "title": work.get("title", "Untitled"),
                    "authors": authors,
                    "first_author_id": first_author_id,
                    "year": work.get("publication_year", 0),
                    "doi": doi,
                    "url": work.get("doi") or work.get("id", ""),
                    "abstract": abstract,
                    "citation_count": work.get("cited_by_count", 0),
                    "field": field,
                    "source": "openalex"
                }
                papers.append(paper)
                
        except Exception as e:
            logger.error(f"[CrossDomain] Field search error for {field}: {e}")
        
        return papers
    
    def search_all_domains(self, query: str, primary_field: str) -> Dict[str, Any]:
        """
        Search primary field and all related fields in parallel.
        Returns papers grouped by field with connection explanations.
        """
        start_time = time.time()
        
        # Generate cross-domain queries
        cross_queries = self.generate_cross_domain_queries(query, primary_field)
        
        all_results = {
            "primary_field": primary_field,
            "primary_query": query,
            "primary_papers": [],
            "cross_domain_results": [],
            "connections": []
        }
        
        # Search primary field first
        logger.info(f"[CrossDomain] Searching primary field: {primary_field}")
        all_results["primary_papers"] = self.search_field(query, primary_field, max_results=10)
        
        # Search cross-domain fields in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            for cq in cross_queries:
                future = executor.submit(
                    self.search_field,
                    cq["query"],
                    cq["field"],
                    5
                )
                futures[future] = cq
            
            for future in as_completed(futures):
                cq = futures[future]
                try:
                    papers = future.result()
                    all_results["cross_domain_results"].append({
                        "field": cq["field"],
                        "query": cq["query"],
                        "reasoning": cq["reasoning"],
                        "papers": papers,
                        "paper_count": len(papers)
                    })
                except Exception as e:
                    logger.error(f"[CrossDomain] Error searching {cq['field']}: {e}")
        
        # Generate connection explanations
        all_results["connections"] = self._generate_connections(
            all_results["primary_papers"],
            all_results["cross_domain_results"],
            primary_field
        )
        
        elapsed = time.time() - start_time
        all_results["search_time_seconds"] = round(elapsed, 2)
        all_results["total_papers"] = (
            len(all_results["primary_papers"]) +
            sum(r["paper_count"] for r in all_results["cross_domain_results"])
        )
        
        logger.info(f"[CrossDomain] Found {all_results['total_papers']} papers across {len(cross_queries) + 1} fields in {elapsed:.2f}s")
        
        return all_results
    
    def _generate_connections(self, primary_papers: List[Dict], 
                               cross_results: List[Dict], 
                               primary_field: str) -> List[Dict[str, Any]]:
        """Generate explanations of how cross-domain papers connect to primary research"""
        connections = []
        
        for cross_result in cross_results:
            if not cross_result["papers"]:
                continue
                
            # Get most cited cross-domain paper
            top_paper = max(cross_result["papers"], key=lambda p: p.get("citation_count", 0))
            
            connection = {
                "from_field": cross_result["field"],
                "to_field": primary_field,
                "reasoning": cross_result["reasoning"],
                "key_paper": {
                    "title": top_paper.get("title", ""),
                    "authors": top_paper.get("authors", ""),
                    "year": top_paper.get("year", 0),
                    "citations": top_paper.get("citation_count", 0),
                    "doi": top_paper.get("doi", "")
                },
                "potential_application": f"Techniques from {cross_result['field']} could be applied to {primary_field} research"
            }
            connections.append(connection)
        
        return connections


# Global instance
cross_domain_engine = CrossDomainEngine()


def search_cross_domain(query: str, primary_field: str) -> Dict[str, Any]:
    """
    Main function to search across domains.
    Returns papers from primary field and related fields with connection explanations.
    """
    return cross_domain_engine.search_all_domains(query, primary_field)


def generate_cross_queries(query: str, primary_field: str) -> List[Dict[str, Any]]:
    """Generate cross-domain search queries"""
    return cross_domain_engine.generate_cross_domain_queries(query, primary_field)


def generate_transfer_mechanism_explanation(
    source_field: str,
    source_technique: str,
    target_field: str,
    target_problem: str
) -> str:
    """
    PHASE 6: Generate a detailed transfer mechanism explanation (100+ words).
    
    Explains HOW to adapt a technique from one domain to another.
    """
    # Get mechanism template
    template_types = ["optimization", "algorithm", "mechanism", "method", "model"]
    
    # Match technique to template type
    technique_lower = source_technique.lower()
    if any(word in technique_lower for word in ["optim", "convergence", "objective"]):
        mechanism_type = "optimization"
    elif any(word in technique_lower for word in ["algorithm", "compute", "process"]):
        mechanism_type = "algorithm"
    elif any(word in technique_lower for word in ["reaction", "pathway", "kinetic"]):
        mechanism_type = "mechanism"
    elif any(word in technique_lower for word in ["experiment", "assay", "measure"]):
        mechanism_type = "method"
    else:
        mechanism_type = "model"
    
    base_template = CrossDomainEngine.TRANSFER_MECHANISMS.get(mechanism_type, 
        "Apply {source_technique} from {source_field} to {target_problem}")
    
    base_explanation = base_template.format(
        source_technique=source_technique,
        source_field=source_field,
        target_problem=target_problem
    )
    
    # Generate detailed 100+ word explanation
    explanation = f"""{base_explanation}.

To implement this transfer:

1. **Identify Core Components**: Extract the fundamental principles of {source_technique} from {source_field} that are domain-independent. Focus on the mathematical formulation, algorithmic structure, or mechanistic understanding that can generalize.

2. **Adapt Domain-Specific Parameters**: Modify the input representations and constraints to match {target_field} requirements. This includes adjusting data formats, units of measurement, and boundary conditions specific to {target_problem}.

3. **Validate Equivalence**: Verify that the adapted technique maintains the essential properties that made it successful in {source_field}. Use benchmark datasets or synthetic experiments from {target_field} to confirm correctness.

4. **Optimize for Target Domain**: Fine-tune hyperparameters and computational efficiency for {target_field} workflows. This may require leveraging domain-specific hardware, libraries, or expert knowledge.

This connection is non-obvious because researchers in {source_field} and {target_field} typically attend different conferences, publish in different journals, and use different vocabularies to describe similar concepts."""

    return explanation


def validate_nonobvious_connection(
    source_field: str,
    target_field: str,
    connection_reasoning: str
) -> Dict[str, Any]:
    """
    PHASE 6: Validate that a cross-domain connection is genuinely non-obvious.
    
    Returns validation result with score and reasoning.
    """
    # Define what makes a connection obvious vs non-obvious
    OBVIOUS_PAIRS = {
        frozenset(["biology", "medicine"]),
        frozenset(["physics", "engineering"]),
        frozenset(["computer_science", "mathematics"]),
        frozenset(["chemistry", "materials_science"]),
    }
    
    # Check if connection is too obvious
    pair = frozenset([source_field.lower(), target_field.lower()])
    is_obvious_pair = pair in OBVIOUS_PAIRS
    
    # Check for specific technique transfer
    has_specific_technique = any(word in connection_reasoning.lower() for word in [
        "algorithm", "method", "technique", "framework", "model", "approach",
        "optimization", "mechanism", "process", "structure"
    ])
    
    # Check for concrete steps
    has_concrete_steps = any(word in connection_reasoning.lower() for word in [
        "step", "first", "then", "adapt", "modify", "apply", "convert", "transform"
    ])
    
    # Calculate non-obvious score (0-100)
    score = 50  # Base score
    
    if not is_obvious_pair:
        score += 20  # Bonus for non-obvious field pair
    else:
        score -= 10  # Penalty for obvious pairing
    
    if has_specific_technique:
        score += 15
    
    if has_concrete_steps:
        score += 15
    
    # Cap score
    score = max(0, min(100, score))
    
    is_valid = score >= 50
    
    why_nonobvious = []
    if not is_obvious_pair:
        why_nonobvious.append(f"{source_field} and {target_field} researchers attend different conferences")
        why_nonobvious.append(f"Different vocabulary and methodological traditions")
    if has_specific_technique:
        why_nonobvious.append("Requires specific domain knowledge to identify transferable technique")
    if has_concrete_steps:
        why_nonobvious.append("Implementation requires non-trivial adaptation steps")
    
    return {
        "is_valid": is_valid,
        "nonobvious_score": score,
        "why_nonobvious": why_nonobvious if why_nonobvious else ["Standard domain connection"],
        "barriers": [
            f"Researchers in {source_field} may not be aware of problems in {target_field}",
            f"Different mathematical formalisms between fields"
        ]
    }


def get_connections_with_transfer_mechanisms(
    query: str,
    primary_field: str
) -> Dict[str, Any]:
    """
    PHASE 6: Main function to get cross-domain connections with full transfer mechanisms.
    
    Returns cross-domain results enhanced with:
    1. Detailed transfer mechanism explanations (100+ words)
    2. Non-obvious validation
    3. Papers from BOTH domains
    """
    # Get base cross-domain results
    base_results = cross_domain_engine.search_all_domains(query, primary_field)
    
    # Enhance each cross-domain result
    enhanced_results = []
    for result in base_results.get("cross_domain_results", []):
        source_field = result.get("field", "unknown")
        
        # Get technique from reasoning
        reasoning = result.get("reasoning", "")
        source_technique = reasoning.split(" - ")[0] if " - " in reasoning else reasoning[:50]
        
        # Generate transfer mechanism
        transfer_mechanism = generate_transfer_mechanism_explanation(
            source_field=source_field,
            source_technique=source_technique,
            target_field=primary_field,
            target_problem=query
        )
        
        # Validate non-obvious connection
        validation = validate_nonobvious_connection(
            source_field=source_field,
            target_field=primary_field,
            connection_reasoning=reasoning
        )
        
        enhanced_results.append({
            **result,
            "transfer_mechanism": transfer_mechanism,
            "validation": validation,
            "source_domain_papers": result.get("papers", [])[:3],
            "target_domain_papers": base_results.get("primary_papers", [])[:3]
        })
    
    return {
        **base_results,
        "cross_domain_results": enhanced_results
    }

