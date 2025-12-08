"""
PostgreSQL Metadata Store

SQLAlchemy-based metadata storage for scientific papers including authors,
citations, keywords, concepts, and their relationships.
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, Float, DateTime, ForeignKey, Index, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json

from src.config import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

Base = declarative_base()


# Database Models

class Paper(Base):
    """Scientific paper"""
    __tablename__ = 'papers'
    
    id = Column(Integer, primary_key=True)
    paper_id = Column(String(255), unique=True, index=True, nullable=False)
    title = Column(Text)
    abstract = Column(Text)
    year = Column(Integer, index=True)
    doi = Column(String(255), index=True)
    url = Column(Text)
    source = Column(String(50))  # arxiv, pubmed, semantic_scholar, openalex
    citations_count = Column(Integer, default=0, index=True)
    field = Column(String(100), index=True)
    subfield = Column(String(100))
    publication_venue = Column(String(255))
    open_access = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    authors = relationship("PaperAuthor", back_populates="paper", cascade="all, delete-orphan")
    keywords = relationship("PaperKeyword", back_populates="paper", cascade="all, delete-orphan")
    concepts = relationship("PaperConcept", back_populates="paper", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_field_subfield', 'field', 'subfield'),
        Index('idx_year_citations', 'year', 'citations_count'),
    )


class Author(Base):
    """Author information"""
    __tablename__ = 'authors'
    
    id = Column(Integer, primary_key=True)
    author_id = Column(String(255), unique=True, index=True)
    name = Column(String(255), index=True, nullable=False)
    affiliation = Column(Text)
    h_index = Column(Integer)
    email = Column(String(255))
    orcid = Column(String(50))
    
    # Relationships
    papers = relationship("PaperAuthor", back_populates="author")


class PaperAuthor(Base):
    """Junction table for papers and authors"""
    __tablename__ = 'paper_authors'
    
    id = Column(Integer, primary_key=True)
    paper_id = Column(Integer, ForeignKey('papers.id', ondelete='CASCADE'), index=True)
    author_id = Column(Integer, ForeignKey('authors.id', ondelete='CASCADE'), index=True)
    author_order = Column(Integer)  # Position in author list
    corresponding_author = Column(Boolean, default=False)
    
    # Relationships
    paper = relationship("Paper", back_populates="authors")
    author = relationship("Author", back_populates="papers")
    
    __table_args__ = (
        Index('idx_paper_author', 'paper_id', 'author_id'),
    )


class Citation(Base):
    """Citation link between papers"""
    __tablename__ = 'citations'
    
    id = Column(Integer, primary_key=True)
    citing_paper_id = Column(Integer, ForeignKey('papers.id', ondelete='CASCADE'), index=True)
    cited_paper_id = Column(Integer, ForeignKey('papers.id', ondelete='CASCADE'), index=True)
    context = Column(Text)  # Text where citation appears
    influential = Column(Boolean, default=False)
    
    __table_args__ = (
        Index('idx_citation_pair', 'citing_paper_id', 'cited_paper_id'),
    )


class Keyword(Base):
    """Research keywords"""
    __tablename__ = 'keywords'
    
    id = Column(Integer, primary_key=True)
    keyword = Column(String(255), unique=True, index=True, nullable=False)
    
    # Relationships
    papers = relationship("PaperKeyword", back_populates="keyword")


class PaperKeyword(Base):
    """Junction table for papers and keywords"""
    __tablename__ = 'paper_keywords'
    
    id = Column(Integer, primary_key=True)
    paper_id = Column(Integer, ForeignKey('papers.id', ondelete='CASCADE'), index=True)
    keyword_id = Column(Integer, ForeignKey('keywords.id', ondelete='CASCADE'), index=True)
    confidence = Column(Float)  # 0.0 to 1.0
    
    # Relationships
    paper = relationship("Paper", back_populates="keywords")
    keyword = relationship("Keyword", back_populates="papers")


class Concept(Base):
    """Research concepts from OpenAlex"""
    __tablename__ = 'concepts'
    
    id = Column(Integer, primary_key=True)
    concept_id = Column(String(255), unique=True, index=True)
    name = Column(String(255), nullable=False)
    level = Column(Integer)  # Hierarchy level
    parent_concept_id = Column(Integer, ForeignKey('concepts.id'))
    
    # Relationships
    papers = relationship("PaperConcept", back_populates="concept")


class PaperConcept(Base):
    """Junction table for papers and concepts"""
    __tablename__ = 'paper_concepts'
    
    id = Column(Integer, primary_key=True)
    paper_id = Column(Integer, ForeignKey('papers.id', ondelete='CASCADE'), index=True)
    concept_id = Column(Integer, ForeignKey('concepts.id', ondelete='CASCADE'), index=True)
    score = Column(Float)  # 0.0 to 1.0
    
    # Relationships
    paper = relationship("Paper", back_populates="concepts")
    concept = relationship("Concept", back_populates="papers")


class IngestionLog(Base):
    """Track data ingestion processes"""
    __tablename__ = 'ingestion_logs'
    
    id = Column(Integer, primary_key=True)
    source = Column(String(50))  # arxiv, pubmed, etc.
    papers_fetched = Column(Integer, default=0)
    papers_processed = Column(Integer, default=0)
    papers_failed = Column(Integer, default=0)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    status = Column(String(50))  # running, completed, failed
    error_message = Column(Text)


class MetadataStore:
    """
    PostgreSQL metadata store for scientific papers
    
    Manages structured paper metadata, authors, citations, keywords,
    and concepts with full relationship tracking.
    """
    
    def __init__(self, connection_string: Optional[str] = None, config: Optional[Settings] = None):
        """
        Initialize metadata store
        
        Args:
            connection_string: PostgreSQL connection string
            config: Application configuration
        """
        if connection_string:
            self.connection_string = connection_string
        elif config:
            self.connection_string = config.database.metadata_db_path
        else:
            self.connection_string = "postgresql://hypothesis_user:password@localhost/papers_metadata"
        
        logger.info(f"Initializing MetadataStore")
        
        # Create engine
        self.engine = create_engine(self.connection_string, echo=False)
        
        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        logger.info("MetadataStore initialized successfully")
    
    def create_tables(self):
        """Create all database tables"""
        logger.info("Creating database tables")
        
        try:
            Base.metadata.create_all(self.engine)
            logger.info("All tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    def add_paper(self, paper_data: Dict, session: Optional[Session] = None) -> Paper:
        """
        Add a paper to the database
        
        Args:
            paper_data: Dictionary with paper information
            session: Optional existing session for transaction
        
        Returns:
            Paper object
        """
        close_session = False
        if session is None:
            session = self.get_session()
            close_session = True
        
        try:
            # Check if paper already exists
            existing = session.query(Paper).filter_by(paper_id=paper_data['paper_id']).first()
            if existing:
                logger.debug(f"Paper {paper_data['paper_id']} already exists")
                return existing
            
            # Create paper
            paper = Paper(
                paper_id=paper_data['paper_id'],
                title=paper_data.get('title'),
                abstract=paper_data.get('abstract'),
                year=paper_data.get('year'),
                doi=paper_data.get('doi'),
                url=paper_data.get('url'),
                source=paper_data.get('source'),
                citations_count=paper_data.get('citations_count', 0),
                field=paper_data.get('field'),
                subfield=paper_data.get('subfield'),
                publication_venue=paper_data.get('publication_venue'),
                open_access=paper_data.get('open_access', False)
            )
            
            session.add(paper)
            session.commit()
            
            logger.debug(f"Added paper: {paper.paper_id}")
            return paper
            
        except Exception as e:
            logger.error(f"Failed to add paper: {e}")
            session.rollback()
            raise
        finally:
            if close_session:
                session.close()
    
    def get_paper(self, paper_id: str) -> Optional[Dict]:
        """
        Retrieve paper with all related data
        
        Args:
            paper_id: Paper ID
        
        Returns:
            Dictionary with paper and related data
        """
        session = self.get_session()
        
        try:
            paper = session.query(Paper).filter_by(paper_id=paper_id).first()
            
            if not paper:
                return None
            
            # Get authors
            authors = []
            for pa in paper.authors:
                authors.append({
                    'id': pa.author.author_id,
                    'name': pa.author.name,
                    'affiliation': pa.author.affiliation,
                    'order': pa.author_order,
                    'corresponding': pa.corresponding_author
                })
            
            # Get keywords
            keywords = [pk.keyword.keyword for pk in paper.keywords]
            
            # Get concepts
            concepts = [{
                'name': pc.concept.name,
                'score': pc.score
            } for pc in paper.concepts]
            
            return {
                'paper_id': paper.paper_id,
                'title': paper.title,
                'abstract': paper.abstract,
                'year': paper.year,
                'doi': paper.doi,
                'url': paper.url,
                'source': paper.source,
                'citations_count': paper.citations_count,
                'field': paper.field,
                'subfield': paper.subfield,
                'publication_venue': paper.publication_venue,
                'open_access': paper.open_access,
                'authors': authors,
                'keywords': keywords,
                'concepts': concepts
            }
            
        finally:
            session.close()
    
    def add_author(self, author_data: Dict, session: Optional[Session] = None) -> Author:
        """Add or update author"""
        close_session = False
        if session is None:
            session = self.get_session()
            close_session = True
        
        try:
            # Check existing
            author_id = author_data.get('author_id') or author_data.get('name')
            existing = session.query(Author).filter_by(author_id=author_id).first()
            
            if existing:
                return existing
            
            author = Author(
                author_id=author_id,
                name=author_data['name'],
                affiliation=author_data.get('affiliation'),
                h_index=author_data.get('h_index'),
                email=author_data.get('email'),
                orcid=author_data.get('orcid')
            )
            
            session.add(author)
            session.commit()
            
            return author
            
        except Exception as e:
            logger.error(f"Failed to add author: {e}")
            session.rollback()
            raise
        finally:
            if close_session:
                session.close()
    
    def link_author_to_paper(
        self,
        paper_id: str,
        author_data: Dict,
        order: int = 0,
        corresponding: bool = False
    ) -> bool:
        """Create paper-author relationship"""
        session = self.get_session()
        
        try:
            # Get paper
            paper = session.query(Paper).filter_by(paper_id=paper_id).first()
            if not paper:
                logger.warning(f"Paper not found: {paper_id}")
                return False
            
            # Get or create author
            author = self.add_author(author_data, session)
            
            # Create link
            link = PaperAuthor(
                paper_id=paper.id,
                author_id=author.id,
                author_order=order,
                corresponding_author=corresponding
            )
            
            session.add(link)
            session.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to link author to paper: {e}")
            session.rollback()
            return False
        finally:
            session.close()
    
    def add_citation(
        self,
        citing_paper_id: str,
        cited_paper_id: str,
        context: Optional[str] = None,
        influential: bool = False
    ) -> bool:
        """Record citation relationship"""
        session = self.get_session()
        
        try:
            # Get papers
            citing = session.query(Paper).filter_by(paper_id=citing_paper_id).first()
            cited = session.query(Paper).filter_by(paper_id=cited_paper_id).first()
            
            if not citing or not cited:
                logger.warning(f"Papers not found for citation")
                return False
            
            # Check if citation already exists
            existing = session.query(Citation).filter_by(
                citing_paper_id=citing.id,
                cited_paper_id=cited.id
            ).first()
            
            if existing:
                return True
            
            citation = Citation(
                citing_paper_id=citing.id,
                cited_paper_id=cited.id,
                context=context,
                influential=influential
            )
            
            session.add(citation)
            session.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add citation: {e}")
            session.rollback()
            return False
        finally:
            session.close()
    
    def get_citations(self, paper_id: str, direction: str = 'citing') -> List[Dict]:
        """
        Get papers that cite this paper or papers this paper cites
        
        Args:
            paper_id: Paper ID
            direction: 'citing' or 'cited'
        
        Returns:
            List of paper dictionaries
        """
        session = self.get_session()
        
        try:
            paper = session.query(Paper).filter_by(paper_id=paper_id).first()
            if not paper:
                return []
            
            if direction == 'citing':
                # Papers that cite this paper
                citations = session.query(Citation).filter_by(cited_paper_id=paper.id).all()
                results = []
                for cit in citations:
                    citing_paper = session.query(Paper).filter_by(id=cit.citing_paper_id).first()
                    if citing_paper:
                        results.append({
                            'paper_id': citing_paper.paper_id,
                            'title': citing_paper.title,
                            'year': citing_paper.year,
                            'influential': cit.influential
                        })
            else:
                # Papers this paper cites
                citations = session.query(Citation).filter_by(citing_paper_id=paper.id).all()
                results = []
                for cit in citations:
                    cited_paper = session.query(Paper).filter_by(id=cit.cited_paper_id).first()
                    if cited_paper:
                        results.append({
                            'paper_id': cited_paper.paper_id,
                            'title': cited_paper.title,
                            'year': cited_paper.year,
                            'influential': cit.influential
                        })
            
            return results
            
        finally:
            session.close()
    
    def add_keywords(self, paper_id: str, keywords: List[str]) -> bool:
        """Tag paper with keywords"""
        session = self.get_session()
        
        try:
            paper = session.query(Paper).filter_by(paper_id=paper_id).first()
            if not paper:
                return False
            
            for kw_text in keywords:
                # Get or create keyword
                keyword = session.query(Keyword).filter_by(keyword=kw_text).first()
                if not keyword:
                    keyword = Keyword(keyword=kw_text)
                    session.add(keyword)
                    session.flush()
                
                # Link to paper
                link = PaperKeyword(
                    paper_id=paper.id,
                    keyword_id=keyword.id,
                    confidence=1.0
                )
                session.add(link)
            
            session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to add keywords: {e}")
            session.rollback()
            return False
        finally:
            session.close()
    
    def search_by_keyword(
        self,
        keywords: List[str],
        operator: str = 'AND'
    ) -> List[Dict]:
        """
        Find papers by keywords
        
        Args:
            keywords: List of keywords to search
            operator: 'AND' or 'OR'
        
        Returns:
            List of matching papers
        """
        session = self.get_session()
        
        try:
            query = session.query(Paper).join(PaperKeyword).join(Keyword)
            
            if operator == 'AND':
                # Papers must have ALL keywords
                for kw in keywords:
                    query = query.filter(Keyword.keyword == kw)
            else:
                # Papers with ANY keyword
                query = query.filter(Keyword.keyword.in_(keywords))
            
            papers = query.distinct().all()
            
            return [{'paper_id': p.paper_id, 'title': p.title, 'year': p.year} for p in papers]
            
        finally:
            session.close()
    
    def search_by_author(self, author_name: str) -> List[Dict]:
        """Find author's papers"""
        session = self.get_session()
        
        try:
            papers = session.query(Paper).join(PaperAuthor).join(Author).filter(
                Author.name.ilike(f"%{author_name}%")
            ).all()
            
            return [{'paper_id': p.paper_id, 'title': p.title, 'year': p.year} for p in papers]
            
        finally:
            session.close()
    
    def search_by_field(
        self,
        field: Optional[str] = None,
        subfield: Optional[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None
    ) -> List[Dict]:
        """Complex search by field and year range"""
        session = self.get_session()
        
        try:
            query = session.query(Paper)
            
            if field:
                query = query.filter(Paper.field == field)
            if subfield:
                query = query.filter(Paper.subfield == subfield)
            if year_from:
                query = query.filter(Paper.year >= year_from)
            if year_to:
                query = query.filter(Paper.year <= year_to)
            
            papers = query.all()
            
            return [{'paper_id': p.paper_id, 'title': p.title, 'year': p.year} for p in papers]
            
        finally:
            session.close()
    
    def get_trending_papers(
        self,
        field: str,
        days: int = 30,
        min_citations: int = 5
    ) -> List[Dict]:
        """Get recently popular papers"""
        session = self.get_session()
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            papers = session.query(Paper).filter(
                Paper.field == field,
                Paper.citations_count >= min_citations,
                Paper.created_at >= cutoff_date
            ).order_by(Paper.citations_count.desc()).limit(50).all()
            
            return [{
                'paper_id': p.paper_id,
                'title': p.title,
                'year': p.year,
                'citations_count': p.citations_count
            } for p in papers]
            
        finally:
            session.close()
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        session = self.get_session()
        
        try:
            total_papers = session.query(func.count(Paper.id)).scalar()
            total_authors = session.query(func.count(Author.id)).scalar()
            total_citations = session.query(func.count(Citation.id)).scalar()
            
            # Papers by field
            field_counts = session.query(
                Paper.field,
                func.count(Paper.id)
            ).group_by(Paper.field).all()
            
            # Papers by year
            year_counts = session.query(
                Paper.year,
                func.count(Paper.id)
            ).group_by(Paper.year).order_by(Paper.year).all()
            
            return {
                'total_papers': total_papers,
                'total_authors': total_authors,
                'total_citations': total_citations,
                'papers_by_field': dict(field_counts),
                'papers_by_year': dict(year_counts)
            }
            
        finally:
            session.close()


# Example usage
if __name__ == "__main__":
    from src.utils.logger import setup_logging
    
    setup_logging(level="INFO")
    
    # Initialize store
    print("\n=== Initializing MetadataStore ===")
    store = MetadataStore("sqlite:///test_metadata.db")  # SQLite for testing
    store.create_tables()
    
    # Add a paper
    print("\n=== Adding Paper ===")
    paper_data = {
        'paper_id': 'test_001',
        'title': 'Example Paper',
        'abstract': 'This is an example paper',
        'year': 2024,
        'field': 'biology',
        'subfield': 'oncology',
        'source': 'arxiv',
        'citations_count': 10
    }
    paper = store.add_paper(paper_data)
    print(f"Added paper: {paper.paper_id}")
    
    # Add author
    print("\n=== Adding Author ===")
    author_data = {
        'name': 'John Doe',
        'affiliation': 'Example University'
    }
    store.link_author_to_paper('test_001', author_data, order=1)
    
    # Add keywords
    print("\n=== Adding Keywords ===")
    store.add_keywords('test_001', ['machine learning', 'cancer', 'oncology'])
    
    # Retrieve paper
    print("\n=== Retrieving Paper ===")
    retrieved = store.get_paper('test_001')
    print(f"Title: {retrieved['title']}")
    print(f"Authors: {[a['name'] for a in retrieved['authors']]}")
    print(f"Keywords: {retrieved['keywords']}")
    
    # Statistics
    print("\n=== Statistics ===")
    stats = store.get_statistics()
    print(f"Total papers: {stats['total_papers']}")
    print(f"Total authors: {stats['total_authors']}")
    
    print("\n✅ All examples completed!")
