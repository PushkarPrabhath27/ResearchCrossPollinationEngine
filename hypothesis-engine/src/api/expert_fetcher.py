"""
Expert Fetcher - Get REAL researcher data
Uses OpenAlex Authors API (FREE, no key required)

Features:
- Extract top authors from fetched papers
- Get author details: h-index, citations, affiliation
- Find relevant collaborators for hypothesis
"""

import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Expert:
    """Real researcher data from OpenAlex"""
    name: str
    openalex_id: str
    orcid: str
    affiliation: str
    affiliation_country: str
    citation_count: int
    works_count: int
    h_index: int
    recent_works: List[str]  # Titles of recent papers
    research_topics: List[str]
    profile_url: str
    relevance_reason: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExpertFetcher:
    """
    Fetches REAL expert/researcher data from OpenAlex.
    
    OpenAlex Authors API:
    - FREE, no API key required
    - 50M+ researchers
    - Has: h-index, citation count, affiliation, ORCID, works
    
    Documentation: https://docs.openalex.org/api-entities/authors
    """
    
    OPENALEX_BASE = "https://api.openalex.org"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ScienceBridge/1.0 (research-discovery; mailto:contact@sciencebridge.ai)"
        })
    
    def get_author_by_id(self, author_id: str) -> Optional[Expert]:
        """
        Get detailed author info from OpenAlex ID.
        OpenAlex ID format: https://openalex.org/A12345678
        """
        try:
            # Clean the ID
            if author_id.startswith("https://openalex.org/"):
                author_id = author_id.replace("https://openalex.org/", "")
            
            url = f"{self.OPENALEX_BASE}/authors/{author_id}"
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            return self._parse_author(data)
            
        except Exception as e:
            logger.warning(f"[ExpertFetcher] Error fetching author {author_id}: {e}")
            return None
    
    def get_top_authors_from_papers(self, papers: List[Dict[str, Any]], max_authors: int = 5) -> List[Expert]:
        """
        Extract unique top authors from a list of papers and get their details.
        Prioritizes authors with most citations.
        """
        # Collect unique author IDs
        author_ids = set()
        for paper in papers:
            author_id = paper.get("first_author_id")
            if author_id:
                author_ids.add(author_id)
        
        if not author_ids:
            logger.warning("[ExpertFetcher] No author IDs found in papers")
            return []
        
        logger.info(f"[ExpertFetcher] Found {len(author_ids)} unique authors, fetching top {max_authors}")
        
        # Fetch author details in parallel
        experts = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self.get_author_by_id, aid): aid
                for aid in list(author_ids)[:max_authors * 2]  # Fetch extra in case some fail
            }
            
            for future in as_completed(futures):
                try:
                    expert = future.result()
                    if expert:
                        experts.append(expert)
                except Exception as e:
                    logger.warning(f"[ExpertFetcher] Author fetch failed: {e}")
        
        # Sort by citation count and return top N
        experts.sort(key=lambda e: e.citation_count, reverse=True)
        
        logger.info(f"[ExpertFetcher] Retrieved {len(experts[:max_authors])} experts")
        return experts[:max_authors]
    
    def search_authors(self, query: str, max_results: int = 5) -> List[Expert]:
        """
        Search for authors by name or topic.
        Uses OpenAlex author search API.
        """
        experts = []
        
        try:
            url = f"{self.OPENALEX_BASE}/authors"
            params = {
                "search": query,
                "per-page": max_results,
                "sort": "cited_by_count:desc"
            }
            
            logger.info(f"[ExpertFetcher] Searching authors: {query}")
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            for author_data in data.get("results", []):
                expert = self._parse_author(author_data, relevance_reason=f"Relevant to: {query}")
                if expert:
                    experts.append(expert)
            
            logger.info(f"[ExpertFetcher] Found {len(experts)} experts")
            
        except Exception as e:
            logger.error(f"[ExpertFetcher] Search error: {e}")
        
        return experts
    
    def _parse_author(self, data: Dict[str, Any], relevance_reason: str = "") -> Optional[Expert]:
        """Parse OpenAlex author response into Expert dataclass"""
        try:
            # Get affiliation
            affiliation = "Unknown"
            affiliation_country = "Unknown"
            affiliations = data.get("affiliations", [])
            if affiliations:
                inst = affiliations[0].get("institution", {})
                affiliation = inst.get("display_name", "Unknown")
                affiliation_country = inst.get("country_code", "Unknown")
            
            # Fallback to last_known_institution
            if affiliation == "Unknown":
                last_inst = data.get("last_known_institution", {})
                if last_inst:
                    affiliation = last_inst.get("display_name", "Unknown")
                    affiliation_country = last_inst.get("country_code", "Unknown")
            
            # Get ORCID
            orcid = ""
            ids = data.get("ids", {})
            if ids and "orcid" in ids:
                orcid = ids["orcid"].replace("https://orcid.org/", "")
            
            # Get recent works titles
            recent_works = []
            for work in data.get("works", [])[:5]:
                if work.get("title"):
                    recent_works.append(work["title"])
            
            # Get research topics
            topics = []
            for concept in data.get("x_concepts", [])[:5]:
                topics.append(concept.get("display_name", ""))
            
            # Get counts
            citation_count = data.get("cited_by_count", 0)
            works_count = data.get("works_count", 0)
            h_index = data.get("summary_stats", {}).get("h_index", 0)
            
            # If no h-index in summary_stats, estimate
            if not h_index and citation_count and works_count:
                h_index = min(int((citation_count / works_count) ** 0.5 * 2), works_count)
            
            expert = Expert(
                name=data.get("display_name", "Unknown"),
                openalex_id=data.get("id", ""),
                orcid=orcid,
                affiliation=affiliation,
                affiliation_country=affiliation_country,
                citation_count=citation_count,
                works_count=works_count,
                h_index=h_index,
                recent_works=recent_works,
                research_topics=topics,
                profile_url=data.get("id", "").replace("https://openalex.org/", "https://openalex.org/authors/"),
                relevance_reason=relevance_reason
            )
            
            return expert
            
        except Exception as e:
            logger.warning(f"[ExpertFetcher] Parse error: {e}")
            return None
    
    def format_experts_for_display(self, experts: List[Expert]) -> List[Dict[str, Any]]:
        """Format experts for frontend display"""
        formatted = []
        for e in experts:
            formatted.append({
                "name": e.name,
                "affiliation": e.affiliation,
                "country": e.affiliation_country,
                "citations": f"{e.citation_count:,}",
                "h_index": e.h_index,
                "works_count": e.works_count,
                "orcid": e.orcid,
                "orcid_url": f"https://orcid.org/{e.orcid}" if e.orcid else None,
                "openalex_url": e.profile_url,
                "research_topics": e.research_topics[:3],
                "recent_papers": e.recent_works[:3],
                "relevance": e.relevance_reason,
                "collaboration_potential": "HIGH" if e.h_index > 30 else ("MEDIUM" if e.h_index > 10 else "ACCESSIBLE")
            })
        return formatted


# Global instance
expert_fetcher = ExpertFetcher()


def get_experts_from_papers(papers: List[Dict[str, Any]], max_experts: int = 5) -> List[Dict[str, Any]]:
    """
    Main function to get REAL experts from a list of papers.
    Uses OpenAlex Authors API (FREE, no key).
    
    Returns formatted expert data ready for display.
    """
    experts = expert_fetcher.get_top_authors_from_papers(papers, max_experts)
    return expert_fetcher.format_experts_for_display(experts)


def search_experts(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search for experts by name or research area"""
    experts = expert_fetcher.search_authors(query, max_results)
    return expert_fetcher.format_experts_for_display(experts)


def generate_estimated_email(name: str, institution: str) -> str:
    """
    PHASE 7: Generate estimated email address from name and institution.
    
    Common patterns:
    - firstname.lastname@institution.edu
    - firstinitial.lastname@institution.edu
    - lastname@institution.edu
    """
    if not name or not institution:
        return None
    
    # Parse name
    parts = name.replace("Dr.", "").replace("Prof.", "").replace(".", "").strip().split()
    if len(parts) < 2:
        return None
    
    first_name = parts[0].lower()
    last_name = parts[-1].lower()
    first_initial = first_name[0] if first_name else ""
    
    # Parse institution to get domain
    institution_lower = institution.lower()
    
    # Common institution domain mappings
    INSTITUTION_DOMAINS = {
        "mit": "mit.edu",
        "stanford": "stanford.edu",
        "harvard": "harvard.edu",
        "berkeley": "berkeley.edu",
        "caltech": "caltech.edu",
        "oxford": "ox.ac.uk",
        "cambridge": "cam.ac.uk",
        "eth": "ethz.ch",
        "max planck": "mpg.de",
        "cnrs": "cnrs.fr",
        "university of": None,  # Will need parsing
    }
    
    # Try to find matching domain
    domain = None
    for key, dom in INSTITUTION_DOMAINS.items():
        if key in institution_lower:
            domain = dom
            break
    
    if not domain:
        # Generate from institution name
        words = [w for w in institution.split() if len(w) > 2 and w.lower() not in ["of", "the", "and", "for"]]
        if words:
            first_word = words[0].lower().replace(",", "")
            domain = f"{first_word}.edu"
    
    if domain:
        # Use most common pattern: firstname.lastname@domain
        return f"{first_name}.{last_name}@{domain}"
    return None


def generate_collaboration_email_template(
    expert_name: str,
    expert_expertise: str,
    query: str,
    contribution: str
) -> Dict[str, str]:
    """
    PHASE 7: Generate ready-to-send collaboration email template.
    
    Returns:
        {
            "subject": "Collaboration Inquiry: ...",
            "body": "Dear Dr. ..., I am writing regarding..."
        }
    """
    # Extract last name for greeting
    name_parts = expert_name.replace("Dr.", "").replace("Prof.", "").strip().split()
    last_name = name_parts[-1] if name_parts else "Researcher"
    
    # Generate subject line
    query_short = query[:50] + "..." if len(query) > 50 else query
    subject = f"Collaboration Inquiry: {query_short}"
    
    # Generate body
    body = f"""Dear Dr. {last_name},

I am writing to inquire about a potential research collaboration in the area of {expert_expertise}.

I am currently working on a project investigating: "{query}"

Based on your expertise in {expert_expertise}, I believe your insights would be invaluable, particularly for {contribution}.

I would be grateful for the opportunity to discuss this further at your convenience. I am happy to share our preliminary findings and discuss how we might collaborate.

Thank you for considering this inquiry.

Best regards,
[Your Name]
[Your Institution]
[Your Email]"""

    return {
        "subject": subject,
        "body": body
    }


def format_experts_for_llm(
    experts: List[Dict[str, Any]], 
    papers: List[Dict[str, Any]], 
    query: str
) -> List[Dict[str, Any]]:
    """
    PHASE 7 ENHANCED: Format experts for LLM prompt with:
    1. why_contact - explanation of what they could contribute
    2. relevant_papers - their papers from the retrieved list
    3. collaboration_likelihood - HIGH/MEDIUM/LOW with reasoning
    4. expertise match to query
    5. NEW: estimated email address
    6. NEW: ready-to-send email template
    """
    formatted = []
    
    # Build paper author lookup
    author_papers = {}
    for paper in papers:
        authors = paper.get('authors', '').lower()
        for expert in experts:
            name = expert.get('name', '').lower()
            # Check if expert authored this paper
            if any(part in authors for part in name.split()[:2] if len(part) > 2):
                if expert['name'] not in author_papers:
                    author_papers[expert['name']] = []
                author_papers[expert['name']].append(paper.get('title', ''))
    
    for expert in experts:
        name = expert.get('name', 'Unknown')
        institution = expert.get('affiliation', 'Unknown')
        h_index = expert.get('h_index', 0)
        topics = expert.get('research_topics', [])
        
        # Generate why_contact based on their expertise
        topics_str = ", ".join(topics[:3]) if topics else "related areas"
        why_contact = f"Expert in {topics_str}. "
        
        # Determine contribution type
        contribution = "validation and expert review"
        if h_index > 30:
            why_contact += f"Highly influential researcher (h-index: {h_index}) who could provide senior guidance and credibility."
            contribution = "senior mentorship and methodological guidance"
            priority = "HIGHEST"
        elif h_index > 15:
            why_contact += f"Established researcher (h-index: {h_index}) with proven track record in this area."
            contribution = "technical expertise and data access"
            priority = "SECONDARY"
        elif h_index > 5:
            why_contact += f"Active researcher (h-index: {h_index}) potentially open to new collaborations."
            contribution = "hands-on collaboration and data generation"
            priority = "SECONDARY"
        else:
            why_contact += f"Early-career researcher who may be highly accessible for collaboration."
            contribution = "active experimental work"
            priority = "FOUNDATIONAL"
        
        # Get their papers from retrieved list
        their_papers = author_papers.get(name, [])
        if not their_papers:
            their_papers = expert.get('recent_papers', [])[:2]
        
        # Generate collaboration likelihood with reasoning
        works_count = expert.get('works_count', 0)
        
        if h_index > 40:
            likelihood = "LOW"
            likelihood_percent = "25%"
            evidence_for = ["Highly recognized in field"]
            evidence_against = ["Very senior, may have limited bandwidth", "Likely receives many requests"]
        elif h_index > 20 and works_count > 100:
            likelihood = "MEDIUM"
            likelihood_percent = "50%"
            evidence_for = ["Active researcher", "Established track record"]
            evidence_against = ["May be focused on existing projects"]
        else:
            likelihood = "HIGH"
            likelihood_percent = "75%"
            evidence_for = ["Actively publishing", "May be seeking collaborations", "Responsive to outreach"]
            evidence_against = ["May have less resources"]
        
        # PHASE 7: Generate estimated email
        estimated_email = generate_estimated_email(name, institution)
        
        # PHASE 7: Generate email template
        email_template = generate_collaboration_email_template(
            name, topics_str, query, contribution
        )
        
        formatted.append({
            "name": name,
            "institution": institution,
            "email": estimated_email,
            "expertise_summary": topics_str,
            "relevant_papers": their_papers[:3],
            "h_index": h_index,
            "citation_count": expert.get('citations', 'N/A'),
            "why_contact": why_contact,
            "contributions": [
                {
                    "contribution_type": "Technical Expertise",
                    "description": contribution,
                    "value_to_project": f"Saves {2 if h_index > 20 else 4} weeks of literature review"
                }
            ],
            "collaboration_likelihood": {
                "likelihood": f"{likelihood} ({likelihood_percent})",
                "evidence_for": evidence_for,
                "evidence_against": evidence_against
            },
            "email_template": email_template,
            "priority": priority,
            "orcid": expert.get('orcid', ''),
            "openalex_url": expert.get('openalex_url', '')
        })
    
    # Sort by priority (HIGHEST first)
    priority_order = {"HIGHEST": 0, "SECONDARY": 1, "FOUNDATIONAL": 2}
    formatted.sort(key=lambda x: priority_order.get(x.get('priority', 'FOUNDATIONAL'), 3))
    
    return formatted


def get_experts_with_why_contact(
    papers: List[Dict[str, Any]], 
    query: str,
    max_experts: int = 5
) -> List[Dict[str, Any]]:
    """
    Get experts from papers with full why_contact reasoning.
    This is the function to use for the new output schema.
    """
    # Get basic expert data
    experts = get_experts_from_papers(papers, max_experts)
    
    # Enhance with why_contact and paper matching
    enhanced = format_experts_for_llm(experts, papers, query)
    
    return enhanced

