"""
Paper Parser

Comprehensive text parser that handles PDF, XML, and HTML formats.
Includes text cleaning, section extraction, and intelligent chunking.
"""

import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from pypdf import PdfReader
from bs4 import BeautifulSoup

from src.config import Settings
from src.utils.logger import get_logger
from src.utils.helpers import clean_text as basic_clean

logger = get_logger(__name__)


class PaperParser:
    """
    Parses scientific papers from multiple formats
    
    Handles PDF, XML (PubMed, arXiv), and HTML. Extracts sections,
    cleans text, and creates semantic chunks for embedding.
    """
    
    # Common section headers
    SECTION_PATTERNS = {
        'abstract': r'^\s*(abstract|summary)\s*$',
        'introduction': r'^\s*(introduction|background)\s*$',
        'methods': r'^\s*(methods?|methodology|materials?\s+and\s+methods?)\s*$',
        'results': r'^\s*(results?|findings?)\s*$',
        'discussion': r'^\s*(discussion)\s*$',
        'conclusion': r'^\s*(conclusions?|summary)\s*$',
        'references': r'^\s*(references?|bibliography|works?\s+cited)\s*$'
    }
    
    def __init__(self, config: Optional[Settings] = None):
        """
        Initialize parser
        
        Args:
            config: Application configuration
        """
        self.config = config
        logger.info("PaperParser initialized")
    
    def parse_pdf(self, file_path: Path) -> str:
        """
        Extract text from PDF
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            Extracted text
        
        Raises:
            IOError: If PDF cannot be read
        """
        if not Path(file_path).exists():
            raise IOError(f"PDF not found: {file_path}")
        
        logger.info(f"Parsing PDF: {file_path}")
        
        try:
            reader = PdfReader(str(file_path))
            text_parts = []
            
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Failed to extract page {page_num}: {e}")
            
            full_text = "\n\n".join(text_parts)
            logger.info(f"Extracted {len(full_text)} characters from {len(reader.pages)} pages")
            
            return full_text
            
        except Exception as e:
            logger.error(f"PDF parsing failed: {e}")
            raise IOError(f"Could not parse PDF: {e}")
    
    def parse_xml(self, xml_content: str, format: str = 'pubmed') -> str:
        """
        Parse XML from various formats
        
        Args:
            xml_content: XML string
            format: XML format ('pubmed', 'arxiv', 'pmc', 'generic')
        
        Returns:
            Extracted text
        """
        logger.debug(f"Parsing {format} XML")
        
        try:
            root = ET.fromstring(xml_content)
            
            if format == 'pubmed':
                return self._parse_pubmed_xml(root)
            elif format == 'pmc':
                return self._parse_pmc_xml(root)
            elif format == 'arxiv':
                return self._parse_arxiv_xml(root)
            else:
                # Generic: extract all text
                return ' '.join(root.itertext())
                
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            return ""
    
    def _parse_pubmed_xml(self, root: ET.Element) -> str:
        """Parse PubMed XML"""
        parts = []
        
        # Title
        title = root.find(".//ArticleTitle")
        if title is not None and title.text:
            parts.append(f"TITLE: {title.text}")
        
        # Abstract
        abstract_parts = root.findall(".//AbstractText")
        if abstract_parts:
            abstract_text = " ".join([a.text for a in abstract_parts if a.text])
            parts.append(f"\nABSTRACT:\n{abstract_text}")
        
        return "\n\n".join(parts)
    
    def _parse_pmc_xml(self, root: ET.Element) -> str:
        """Parse PMC full-text XML"""
        parts = []
        
        # Extract sections from body
        body = root.find(".//body")
        if body is not None:
            for section in body.findall(".//sec"):
                title_elem = section.find(".//title")
                if title_elem is not None and title_elem.text:
                    parts.append(f"\n{title_elem.text.upper()}")
                
                # Get paragraphs
                for para in section.findall(".//p"):
                    if para.text:
                        parts.append(para.text)
        
        return "\n\n".join(parts)
    
    def _parse_arxiv_xml(self, root: ET.Element) -> str:
        """Parse arXiv XML"""
        parts = []
        
        # Title
        title = root.find(".//{http://www.w3.org/2005/Atom}title")
        if title is not None and title.text:
            parts.append(f"TITLE: {title.text}")
        
        # Summary (abstract)
        summary = root.find(".//{http://www.w3.org/2005/Atom}summary")
        if summary is not None and summary.text:
            parts.append(f"\nABSTRACT:\n{summary.text}")
        
        return "\n\n".join(parts)
    
    def parse_html(self, html_content: str) -> str:
        """
        Parse HTML papers
        
        Args:
            html_content: HTML string
        
        Returns:
            Extracted text
        """
        logger.debug("Parsing HTML")
        
        try:
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Remove script and style elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Extract text
            text = soup.get_text(separator='\n')
            
            return text
            
        except Exception as e:
            logger.error(f"HTML parsing error: {e}")
            return ""
    
    def clean_text(self, raw_text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            raw_text: Raw extracted text
        
        Returns:
            Cleaned text
        """
        logger.debug("Cleaning text")
        
        text = raw_text
        
        # Remove page numbers (e.g., "Page 1", "- 5 -")
        text = re.sub(r'^\s*-?\s*\d+\s*-?\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*Page\s+\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Fix hyphenation across line breaks
        text = re.sub(r'(\w+)-\s+\n\s*(\w+)', r'\1\2', text)
        
        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Fix common LaTeX artifacts
        text = re.sub(r'\\[a-z]+\{([^}]*)\}', r'\1', text)  # \textbf{text} -> text
        text = re.sub(r'\\[a-z]+', '', text)  # Remove LaTeX commands
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{4,}', '...', text)
        
        # Clean up
        text = text.strip()
        
        return text
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Identify and extract paper sections
        
        Args:
            text: Full paper text
        
        Returns:
            Dictionary mapping section names to content
        """
        logger.debug("Extracting sections")
        
        sections = {}
        current_section = 'introduction'
        current_content = []
        
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if this is a section header
            section_found = None
            for section_name, pattern in self.SECTION_PATTERNS.items():
                if re.match(pattern, line_lower, re.IGNORECASE):
                    # Save previous section
                    if current_content:
                        sections[current_section] = '\n'.join(current_content).strip()
                    
                    # Start new section
                    section_found = section_name
                    current_section = section_name
                    current_content = []
                    break
            
            if not section_found:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        logger.info(f"Extracted {len(sections)} sections")
        return sections
    
    def chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50,
        preserve_sections: bool = True
    ) -> List[Dict]:
        """
        Split text into chunks with overlap
        
        Args:
            text: Text to chunk
            chunk_size: Target size in words
            overlap: Overlap size in words
            preserve_sections: Whether to keep section labels
        
        Returns:
            List of chunk dictionaries with metadata
        """
        logger.debug(f"Chunking text: chunk_size={chunk_size}, overlap={overlap}")
        
        chunks = []
        
        if preserve_sections:
            # Extract sections first
            sections = self.extract_sections(text)
            
            for section_name, section_text in sections.items():
                section_chunks = self._chunk_single_section(
                    section_text,
                    section_name,
                    chunk_size,
                    overlap
                )
                chunks.extend(section_chunks)
        else:
            # Chunk entire text
            chunks = self._chunk_single_section(text, 'full_text', chunk_size, overlap)
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def _chunk_single_section(
        self,
        text: str,
        section_name: str,
        chunk_size: int,
        overlap: int
    ) -> List[Dict]:
        """Chunk a single section"""
        chunks = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        current_chunk = []
        current_words = 0
        chunk_index = 0
        
        for sentence in sentences:
            words = sentence.split()
            sentence_words = len(words)
            
            if current_words + sentence_words > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'section': section_name,
                    'index': chunk_index,
                    'word_count': current_words,
                    'start_pos': chunk_index * (chunk_size - overlap),
                    'end_pos': chunk_index * (chunk_size - overlap) + current_words
                })
                
                # Start new chunk with overlap
                overlap_words = []
                overlap_count = 0
                for sent in reversed(current_chunk):
                    sent_words = len(sent.split())
                    if overlap_count + sent_words <= overlap:
                        overlap_words.insert(0, sent)
                        overlap_count += sent_words
                    else:
                        break
                
                current_chunk = overlap_words
                current_words = overlap_count
                chunk_index += 1
            
            current_chunk.append(sentence)
            current_words += sentence_words
        
        # Save last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'section': section_name,
                'index': chunk_index,
                'word_count': current_words,
                'start_pos': chunk_index * (chunk_size - overlap),
                'end_pos': chunk_index * (chunk_size - overlap) + current_words
            })
        
        return chunks
    
    def extract_references(self, text: str) -> List[str]:
        """
        Extract reference list from text
        
        Args:
            text: Paper text
        
        Returns:
            List of reference strings
        """
        logger.debug("Extracting references")
        
        # Find references section
        refs_match = re.search(
            r'(?:^|\n)\s*(?:references?|bibliography|works?\s+cited)\s*\n(.+)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        
        if not refs_match:
            return []
        
        refs_text = refs_match.group(1)
        
        # Split into individual references
        # Common patterns: [1], 1., (1)
        references = re.split(r'\n\s*[\[\(]?\d+[\]\).]?\s+', refs_text)
        references = [ref.strip() for ref in references if ref.strip()]
        
        logger.info(f"Extracted {len(references)} references")
        return references
    
    def extract_figures_tables(self, content: str) -> Dict[str, List[Dict]]:
        """
        Extract figure and table captions
        
        Args:
            content: Paper content
        
        Returns:
            Dictionary with 'figures' and 'tables' lists
        """
        logger.debug("Extracting figures and tables")
        
        result = {'figures': [], 'tables': []}
        
        # Extract figures
        fig_pattern = r'Figure\s+(\d+)[:\.]?\s*(.+?)(?=\n\n|Figure\s+\d+|\Z)'
        figures = re.findall(fig_pattern, content, re.IGNORECASE | re.DOTALL)
        
        for fig_num, caption in figures:
            result['figures'].append({
                'number': fig_num,
                'caption': caption.strip()
            })
        
        # Extract tables
        table_pattern = r'Table\s+(\d+)[:\.]?\s*(.+?)(?=\n\n|Table\s+\d+|\Z)'
        tables = re.findall(table_pattern, content, re.IGNORECASE | re.DOTALL)
        
        for table_num, caption in tables:
            result['tables'].append({
                'number': table_num,
                'caption': caption.strip()
            })
        
        logger.info(f"Extracted {len(result['figures'])} figures, {len(result['tables'])} tables")
        return result


# Example usage
if __name__ == "__main__":
    from src.utils.logger import setup_logging
    
    setup_logging(level="INFO")
    parser = PaperParser()
    
    # Example text
    sample_text = """
    TITLE: Example Scientific Paper
    
    ABSTRACT
    This is the abstract of the paper. It summarizes the main findings.
    
    INTRODUCTION
    Scientific research is important. This paper investigates X.
    We propose a novel approach based on Y.
    
    METHODS
    We collected data from Z sources. Analysis was performed using ABC method.
    Statistical significance was tested using t-tests.
    
    RESULTS
    We found significant effects. Figure 1 shows the main results.
    Table 1 lists all measurements.
    
    DISCUSSION
    These results suggest that our hypothesis is correct.
    Previous work by Smith et al. supports this finding.
    
    CONCLUSION
    In conclusion, we have demonstrated X leads to Y.
    
    REFERENCES
    1. Smith J. et al. (2020). Previous work. Nature.
    2. Jones A. (2021). Related study. Science.
    """
    
    # Clean text
    print("\n=== Cleaning Text ===")
    cleaned = parser.clean_text(sample_text)
    print(f"Cleaned {len(cleaned)} characters")
    
    # Extract sections
    print("\n=== Extracting Sections ===")
    sections = parser.extract_sections(cleaned)
    for section, content in sections.items():
        print(f"{section}: {len(content)} chars")
    
    # Chunk text
    print("\n=== Chunking Text ===")
    chunks = parser.chunk_text(cleaned, chunk_size=50, overlap=10)
    print(f"Created {len(chunks)} chunks:")
    for chunk in chunks[:3]:
        print(f"  Chunk {chunk['index']} ({chunk['section']}): {chunk['word_count']} words")
    
    # Extract references
    print("\n=== Extracting References ===")
    refs = parser.extract_references(cleaned)
    print(f"Found {len(refs)} references")
    
    # Extract figures/tables
    print("\n=== Extracting Figures/Tables ===")
    figs_tables = parser.extract_figures_tables(cleaned)
    print(f"Figures: {len(figs_tables['figures'])}, Tables: {len(figs_tables['tables'])}")
    
    print("\n✅ All examples completed!")
