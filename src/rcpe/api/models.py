"""
Data Models for ScienceBridge Complete Hypothesis Output

These models match the EXACT JSON schema from updatesprompt.md.
Every field is designed to enforce:
1. Citation quality (papers must exist in retrieved list)
2. Specificity (numbers required)
3. Mechanism explanation (HOW not just WHAT)
4. Cross-domain connections (genuine technique transfer)
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


# ==================== PROBLEM CONTEXT ====================

class CurrentSOTA(BaseModel):
    """Current state-of-the-art method with numbers"""
    method: str = Field(..., description="Name of best current method")
    performance: str = Field(..., description="Metric with numbers (e.g., '90% in 10 hours at 72°C')")
    source: str = Field(..., description="Full citation: Author (Year) - Paper Title - DOI")
    limitation: str = Field(..., description="Specific weakness with numbers")


class FailedAttempt(BaseModel):
    """Previously tried approach that didn't work"""
    approach: str = Field(..., description="What was tried")
    researchers: str = Field(..., description="Who tried it (Author et al. Year)")
    result: str = Field(..., description="What happened with numbers")
    why_failed: str = Field(..., description="Root cause of failure")
    source: str = Field(..., description="Paper citation with DOI")


class ProblemContext(BaseModel):
    """Literature gap analysis showing SOTA, failures, and unmet need"""
    current_sota: CurrentSOTA
    failed_attempts: List[FailedAttempt] = Field(default_factory=list)
    unmet_need: str = Field(..., description="Specific gap with quantified impact")


# ==================== HYPOTHESIS ====================

class HypothesisCore(BaseModel):
    """The core hypothesis with theoretical foundation"""
    main_claim: str = Field(..., description="Clear 2-3 sentence statement of proposal")
    theoretical_basis: str = Field(..., description="Mechanism explanation citing 3-5 papers with specific findings")
    novelty: str = Field(..., description="What has NOT been done before + why this is new")
    expected_improvement: str = Field(..., description="Quantitative prediction vs SOTA (e.g., '10x faster')")


# ==================== CROSS-DOMAIN ====================

class CrossDomainConnection(BaseModel):
    """Genuine technique transfer between fields"""
    source_domain: str = Field(..., description="Field A where technique comes from")
    source_technique: str = Field(..., description="Specific method/technique name")
    source_paper: str = Field(..., description="Full citation with DOI")
    source_finding: str = Field(..., description="What they found with numbers")
    target_domain: str = Field(..., description="Field B where problem exists")
    target_problem: str = Field(..., description="The problem being solved")
    transfer_mechanism: str = Field(..., description="HOW to adapt technique A for problem B (specific steps)")
    why_nonobvious: str = Field(..., description="Why experts haven't connected these before")


# ==================== METHODOLOGY ====================

class MethodologyStep(BaseModel):
    """Detailed implementation step with sources"""
    step_number: int
    step_name: str = Field(..., description="What you're doing")
    algorithm: str = Field(..., description="Specific algorithm name and version (e.g., 'Rosetta FastRelax v3.12')")
    parameters: Dict[str, str] = Field(
        default_factory=dict, 
        description="Parameter: value (justification with citation)"
    )
    source_papers: List[str] = Field(
        default_factory=list,
        description="Citations for each decision"
    )
    input_spec: str = Field(..., description="What goes in (format, size, source)")
    output_spec: str = Field(..., description="What comes out (expected values with numbers)")
    success_criteria: str = Field(..., description="How to know if it worked (threshold)")
    time_estimate: str = Field(..., description="X days/weeks with breakdown")
    resources_needed: str = Field(..., description="Compute/equipment/reagents with costs")
    code_snippet: Optional[str] = Field(None, description="Brief working code if applicable")


# ==================== COMPARISON TABLE ====================

class ComparisonMethod(BaseModel):
    """Existing method for comparison"""
    name: str
    performance: str = Field(..., description="Number with unit from literature")
    cost: str = Field(..., description="$/unit estimate")
    advantages: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    source: Optional[str] = Field(None, description="Citation if existing method")


class ComparisonTable(BaseModel):
    """Side-by-side comparison with existing methods"""
    methods: List[ComparisonMethod] = Field(default_factory=list)
    recommendation: str = Field(..., description="When to use each method")


# ==================== EXPERTS ====================

class ExpertCollaborator(BaseModel):
    """Real researcher to contact for collaboration"""
    name: str = Field(..., description="Dr. First Last")
    institution: str
    email: Optional[str] = None
    expertise: str = Field(..., description="Specific area")
    relevant_papers: List[str] = Field(default_factory=list, description="Their papers from retrieved list")
    h_index: Optional[int] = None
    citation_count: Optional[int] = None
    why_contact: str = Field(..., description="What they could contribute to this project")
    collaboration_likelihood: str = Field(..., description="HIGH/MEDIUM/LOW with reasoning")


# ==================== RISK ASSESSMENT ====================

class RiskLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class RiskAssessment(BaseModel):
    """Project risk with probability and mitigation"""
    risk: str = Field(..., description="What could go wrong")
    probability: str = Field(..., description="% with justification (e.g., '60% based on Austin et al.')")
    impact: RiskLevel
    evidence: str = Field(..., description="Citation showing others hit this issue")
    mitigation: str = Field(..., description="How to reduce risk")
    contingency: str = Field(..., description="Backup plan if it happens")


# ==================== VALIDATION METRICS ====================

class ValidationMetric(BaseModel):
    """Success metric with SOTA comparison"""
    metric: str = Field(..., description="What you'll measure")
    current_sota: str = Field(..., description="Best existing value with citation")
    your_target: str = Field(..., description="What you're aiming for")
    measurement_method: str = Field(..., description="How you'll measure with citation")
    success_threshold: str = Field(..., description="Minimum to be publishable")


class ValidationMetrics(BaseModel):
    """All validation metrics organized"""
    primary_metrics: List[ValidationMetric] = Field(default_factory=list)
    secondary_metrics: List[ValidationMetric] = Field(default_factory=list)


# ==================== DATASETS & CODE ====================

class RelevantDataset(BaseModel):
    """Dataset with WHY it's relevant"""
    name: str
    source: str
    url: str
    size: str
    format: str
    relevance: str = Field(..., description="HOW it helps your research specifically")
    specific_use: str = Field(..., description="What you'll do with it")


class RelevantCode(BaseModel):
    """Code repository with HOW it helps"""
    repo_name: str
    url: str
    language: str
    stars: int
    relevance: str = Field(..., description="HOW it helps your research")
    specific_use: str = Field(..., description="What you'll adapt from it")
    citation: Optional[str] = Field(None, description="Associated paper if any")


# ==================== QUALITY SCORE ====================

class QualityScore(BaseModel):
    """Quality metrics for the hypothesis (X/10)"""
    citation_quality: float = Field(..., ge=0, le=10, description="Papers cited vs fabricated")
    specificity: float = Field(..., ge=0, le=10, description="Concrete numbers included")
    cross_domain: float = Field(..., ge=0, le=10, description="Genuine technique transfers")
    actionability: float = Field(..., ge=0, le=10, description="Clear next steps")
    overall: float = Field(..., ge=0, le=10, description="Average score")
    issues: List[str] = Field(default_factory=list, description="Quality issues found")


# ==================== NEW: EXECUTIVE SUMMARY (ENHANCED) ====================

class ExecutiveSummaryNew(BaseModel):
    """Enhanced executive summary from updated updatesprompt.md"""
    one_sentence: str = Field(..., description="Problem + Solution + Impact in <50 words")
    target_audience: str = Field(..., description="Who should care about this")
    key_innovation: str = Field(..., description="What's novel in <30 words")


# ==================== NEW: PROBLEM SCALE ====================

class ProblemScale(BaseModel):
    """Quantitative scale of the problem"""
    description: str = Field(..., description="How big is the problem")
    quantitative_impact: str = Field(..., description="Numbers showing severity")
    source_papers: List[str] = Field(default_factory=list, description="Citations with full metadata")


# ==================== NEW: NOVELTY ANALYSIS ====================

class LiteratureSearch(BaseModel):
    """Evidence of literature search for novelty"""
    query_used: str = Field(..., description="Search terms used")
    papers_found: int = Field(..., description="Number of papers found")
    closest_work: str = Field(..., description="Most similar paper and how ours differs")


class NoveltyAnalysis(BaseModel):
    """Section A: Why this hasn't been done before"""
    what_has_not_been_done: str = Field(..., description="Specific combination/approach that's new")
    why_not_done_before: str = Field(..., description="Barrier that prevented it")
    why_possible_now: str = Field(..., description="What changed recently to make this possible")
    literature_search: Optional[LiteratureSearch] = None


class WhyNotDoneBefore(BaseModel):
    """Missing Section A: Complete explanation of timing"""
    temporal_barrier: str = Field(..., description="When biological/domain knowledge became available")
    technical_barrier: str = Field(..., description="When tools/methods became available")
    cultural_barrier: str = Field(..., description="Why fields didn't connect before")
    data_barrier: str = Field(..., description="When sufficient data became available")
    opportunity_now: str = Field(..., description="Why timing is perfect NOW")


# ==================== NEW: ALTERNATIVE APPROACHES (SECTION B) ====================

class AlternativeApproach(BaseModel):
    """Missing Section B: Alternative we considered and rejected"""
    option_name: str = Field(..., description="Name of alternative approach")
    what_is_it: str = Field(..., description="Brief description")
    pros: List[str] = Field(default_factory=list, description="Advantages of this approach")
    cons: List[str] = Field(default_factory=list, description="Disadvantages of this approach")
    why_rejected: str = Field(..., description="Specific reason we chose not to use this")


class AlternativesRejected(BaseModel):
    """Collection of rejected alternatives with reasoning"""
    alternatives: List[AlternativeApproach] = Field(default_factory=list)
    why_our_approach_best: str = Field(..., description="Summary of why we chose our approach")


# ==================== NEW: PRELIMINARY DATA (SECTION C) ====================

class PilotStudy(BaseModel):
    """Missing Section C: Preliminary data / proof of concept"""
    study_name: str = Field(..., description="Name of pilot study")
    date: str = Field(..., description="When conducted")
    method: str = Field(..., description="What was done")
    results: List[str] = Field(default_factory=list, description="Key findings with numbers")
    conclusion: str = Field(..., description="What we learned")
    next_steps: str = Field(..., description="What to do next")


class PreliminaryData(BaseModel):
    """Collection of pilot studies and preliminary work"""
    pilot_studies: List[PilotStudy] = Field(default_factory=list)
    overall_readiness: str = Field(..., description="How ready are we to proceed")


# ==================== NEW: BROADER IMPACT (SECTION D) ====================

class ImpactArea(BaseModel):
    """Single impact area with quantification"""
    area: str = Field(..., description="Type of impact (clinical, economic, etc.)")
    description: str = Field(..., description="How this work impacts this area")
    quantitative_benefit: str = Field(..., description="Numbers showing impact")
    source: Optional[str] = Field(None, description="Citation if from literature")


class SuccessMetrics(BaseModel):
    """Short, medium, and long-term success metrics"""
    short_term: List[str] = Field(default_factory=list, description="1-2 year goals")
    medium_term: List[str] = Field(default_factory=list, description="3-5 year goals")
    long_term: List[str] = Field(default_factory=list, description="5-10 year goals")


class BroaderImpact(BaseModel):
    """Missing Section D: Complete broader impact assessment"""
    clinical_impact: Optional[ImpactArea] = None
    economic_impact: Optional[ImpactArea] = None
    scientific_impact: Optional[ImpactArea] = None
    educational_impact: Optional[ImpactArea] = None
    environmental_impact: Optional[ImpactArea] = None
    sdg_alignment: List[str] = Field(default_factory=list, description="UN SDGs this work supports")
    success_metrics: Optional[SuccessMetrics] = None


# ==================== NEW: FUNDING OPPORTUNITIES (SECTION E) ====================

class FundingOpportunity(BaseModel):
    """Missing Section E: Specific funding opportunity"""
    agency: str = Field(..., description="Funding agency name")
    program: str = Field(..., description="Specific program name")
    amount: str = Field(..., description="Dollar amount available")
    duration: str = Field(..., description="Grant duration")
    deadline: str = Field(..., description="Application deadline")
    fit_score: int = Field(..., ge=1, le=10, description="1-10 how well project fits")
    why_good_fit: List[str] = Field(default_factory=list, description="Reasons this is a good match")
    success_rate: str = Field(..., description="Historical success rate %")
    application_strategy: str = Field(..., description="How to approach application")
    recent_examples: List[str] = Field(default_factory=list, description="Similar funded projects")


class FundingPlan(BaseModel):
    """Complete funding strategy"""
    opportunities: List[FundingOpportunity] = Field(default_factory=list)
    application_timeline: str = Field(..., description="When to apply for each")
    total_potential: str = Field(..., description="Total $ if all funded")
    recommended_first: str = Field(..., description="Which to apply to first and why")


# ==================== NEW: IP LANDSCAPE (SECTION F) ====================

class Patent(BaseModel):
    """Missing Section F: Patent in the IP landscape"""
    patent_number: str = Field(..., description="Patent number (e.g., US10861139)")
    title: str = Field(..., description="Patent title")
    assignee: str = Field(..., description="Who owns it")
    filed_year: int
    status: str = Field(..., description="Active, Pending, Expired")
    key_claims: List[str] = Field(default_factory=list, description="Relevant claims")
    potential_conflict: str = Field(..., description="What might conflict with our work")
    risk_level: RiskLevel
    risk_reasoning: str = Field(..., description="Why this risk level")
    mitigation: str = Field(..., description="How to avoid conflict")


class IPStrategy(BaseModel):
    """IP strategy options"""
    option: str = Field(..., description="Open Science / Provisional / Full Patent")
    pros: List[str] = Field(default_factory=list)
    cons: List[str] = Field(default_factory=list)
    best_for: str = Field(..., description="Who should choose this option")


class IPLandscape(BaseModel):
    """Complete IP landscape analysis"""
    search_terms: List[str] = Field(default_factory=list, description="What we searched for")
    patents_found: int = Field(..., description="Total patents found")
    relevant_patents: List[Patent] = Field(default_factory=list)
    strategies: List[IPStrategy] = Field(default_factory=list)
    recommendation: str = Field(..., description="Recommended IP approach")
    estimated_cost: str = Field(..., description="Cost of recommended approach")


# ==================== NEW: ENHANCED EXPERT WITH EMAIL TEMPLATE ====================

class CollaborationEmailTemplate(BaseModel):
    """Ready-to-send collaboration request email"""
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Full email body text")


class ExpertContribution(BaseModel):
    """Specific contribution an expert could make"""
    contribution_type: str = Field(..., description="Data Sharing, Validation, Consultation, etc.")
    description: str = Field(..., description="What exactly they could contribute")
    value_to_project: str = Field(..., description="How this helps (with estimate if possible)")


class CollaborationLikelihood(BaseModel):
    """Evidence-based collaboration likelihood"""
    likelihood: str = Field(..., description="VERY HIGH/HIGH/MEDIUM/LOW with %")
    evidence_for: List[str] = Field(default_factory=list, description="Why they might accept")
    evidence_against: List[str] = Field(default_factory=list, description="Why they might decline")


class EnhancedExpertCollaborator(BaseModel):
    """Complete expert profile with email template"""
    name: str = Field(..., description="Dr. First Last")
    institution: str
    position: Optional[str] = None
    email: Optional[str] = None
    lab_website: Optional[str] = None
    linkedin: Optional[str] = None
    relevant_papers: List[str] = Field(default_factory=list, description="Their papers from retrieved list")
    h_index: Optional[int] = None
    citation_count: Optional[int] = None
    expertise_summary: str = Field(..., description="What they're expert in")
    contributions: List[ExpertContribution] = Field(default_factory=list)
    collaboration_likelihood: Optional[CollaborationLikelihood] = None
    email_template: Optional[CollaborationEmailTemplate] = None
    priority: str = Field(..., description="HIGHEST/SECONDARY/FOUNDATIONAL")


# ==================== NEW: QUALITY CHECKS (SELF-ASSESSMENT) ====================

class QualityChecks(BaseModel):
    """Self-assessment of output quality per 6 rules"""
    all_citations_verified: bool = Field(..., description="Every citation is in RETRIEVED_PAPERS")
    fabricated_citations: List[str] = Field(default_factory=list, description="Any citations NOT in retrieved set")
    top3_papers_used: bool = Field(..., description="Top 3 cited papers appear in hypothesis")
    top3_papers_list: List[str] = Field(default_factory=list, description="What the top 3 papers are")
    vague_words_found: List[str] = Field(default_factory=list, description="Banned words found in output")
    numbers_count: int = Field(..., description="Count of specific numbers in output")
    cross_domain_has_both_citations: bool = Field(..., description="Cross-domain cites both domains")
    methodology_has_code: bool = Field(..., description="Methodology includes code snippets")
    experts_have_email: bool = Field(..., description="Experts have contact info")
    overall_compliance: float = Field(..., ge=0, le=100, description="% of rules followed")


# ==================== COMPLETE OUTPUT (UPDATED) ====================

class CompleteHypothesisOutputNew(BaseModel):
    """
    Complete hypothesis output matching UPDATED updatesprompt.md schema.
    Includes all 6 missing sections (A-F) and enhanced fields.
    """
    # Core (Enhanced)
    hypothesis_title: str = Field(..., description="Descriptive title with key innovation")
    executive_summary: Optional[str] = Field(None, description="Legacy field")
    executive_summary_new: Optional[ExecutiveSummaryNew] = None
    
    # Problem & Solution (Enhanced)
    problem_context: ProblemContext
    problem_scale: Optional[ProblemScale] = None
    hypothesis: HypothesisCore
    novelty_analysis: Optional[NoveltyAnalysis] = None
    
    # Innovation
    cross_domain_connections: List[CrossDomainConnection] = Field(default_factory=list)
    
    # Implementation
    methodology: List[MethodologyStep] = Field(default_factory=list)
    comparison_table: ComparisonTable
    
    # Collaboration (Enhanced)
    expert_collaborators: List[ExpertCollaborator] = Field(default_factory=list)
    enhanced_experts: List[EnhancedExpertCollaborator] = Field(default_factory=list)
    
    # Risk & Validation
    risk_assessment: List[RiskAssessment] = Field(default_factory=list)
    validation_metrics: ValidationMetrics
    
    # Resources
    relevant_datasets: List[RelevantDataset] = Field(default_factory=list)
    relevant_code: List[RelevantCode] = Field(default_factory=list)
    
    # ===== NEW SECTIONS A-F =====
    # Section A: Why This Hasn't Been Done Before
    why_not_done_before: Optional[WhyNotDoneBefore] = None
    
    # Section B: Alternative Approaches Rejected
    alternatives_rejected: Optional[AlternativesRejected] = None
    
    # Section C: Preliminary Data
    preliminary_data: Optional[PreliminaryData] = None
    
    # Section D: Broader Impact
    broader_impact: Optional[BroaderImpact] = None
    
    # Section E: Funding Opportunities
    funding_plan: Optional[FundingPlan] = None
    
    # Section F: IP Landscape
    ip_landscape: Optional[IPLandscape] = None
    
    # Quality (Enhanced)
    quality_score: Optional[QualityScore] = None
    quality_checks: Optional[QualityChecks] = None
    
    # Metadata
    papers_used: List[str] = Field(default_factory=list, description="DOIs of papers actually cited")
    papers_retrieved: List[str] = Field(default_factory=list, description="All retrieved paper DOIs for validation")
    generation_time: Optional[float] = None


# Keep original for backward compatibility
class CompleteHypothesisOutput(BaseModel):
    """
    Complete hypothesis output matching updatesprompt.md schema.
    Every field enforces quality requirements.
    """
    # Core
    hypothesis_title: str = Field(..., description="Descriptive title")
    executive_summary: str = Field(..., description="2-3 sentence summary for non-experts")
    
    # Problem & Solution
    problem_context: ProblemContext
    hypothesis: HypothesisCore
    
    # Innovation
    cross_domain_connections: List[CrossDomainConnection] = Field(default_factory=list)
    
    # Implementation
    methodology: List[MethodologyStep] = Field(default_factory=list)
    comparison_table: ComparisonTable
    
    # Collaboration
    expert_collaborators: List[ExpertCollaborator] = Field(default_factory=list)
    
    # Risk & Validation
    risk_assessment: List[RiskAssessment] = Field(default_factory=list)
    validation_metrics: ValidationMetrics
    
    # Resources
    relevant_datasets: List[RelevantDataset] = Field(default_factory=list)
    relevant_code: List[RelevantCode] = Field(default_factory=list)
    
    # Quality
    quality_score: Optional[QualityScore] = None
    
    # Metadata
    papers_used: List[str] = Field(default_factory=list, description="DOIs of papers actually cited")
    generation_time: Optional[float] = None


# ==================== HELPER FUNCTIONS ====================

def create_empty_hypothesis() -> CompleteHypothesisOutput:
    """Create empty hypothesis structure for error cases"""
    return CompleteHypothesisOutput(
        hypothesis_title="Generation Failed",
        executive_summary="Unable to generate hypothesis",
        problem_context=ProblemContext(
            current_sota=CurrentSOTA(
                method="Unknown",
                performance="N/A",
                source="N/A",
                limitation="N/A"
            ),
            unmet_need="Unable to analyze"
        ),
        hypothesis=HypothesisCore(
            main_claim="Unable to generate",
            theoretical_basis="N/A",
            novelty="N/A",
            expected_improvement="N/A"
        ),
        comparison_table=ComparisonTable(
            methods=[],
            recommendation="N/A"
        ),
        validation_metrics=ValidationMetrics()
    )
