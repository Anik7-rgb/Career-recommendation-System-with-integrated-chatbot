# === resume_parser.py ===
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span
import re
import fitz  # PyMuPDF
from typing import Dict, List, Set, Optional
import docx
from pathlib import Path

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# ==================== COMPREHENSIVE SKILL TAXONOMY ====================
SKILL_DATABASE = {
    # Programming Languages
    "languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "c", "ruby", 
        "php", "swift", "kotlin", "go", "rust", "scala", "r", "matlab", "perl",
        "objective-c", "dart", "elixir", "haskell", "lua", "assembly", "fortran"
    ],
    # Web Technologies
    "web": [
        "html", "html5", "css", "css3", "sass", "less", "react", "angular", "vue.js",
        "vue", "next.js", "nuxt.js", "svelte", "node.js", "express.js", "django",
        "flask", "fastapi", "spring boot", "asp.net", "laravel", "ruby on rails",
        "jquery", "bootstrap", "tailwind css", "webpack", "babel"
    ],
    # Data Science & ML
    "data_science": [
        "machine learning", "deep learning", "neural networks", "nlp", 
        "natural language processing", "computer vision", "tensorflow", "pytorch",
        "keras", "scikit-learn", "sklearn", "pandas", "numpy", "scipy", "matplotlib",
        "seaborn", "plotly", "jupyter", "opencv", "hugging face", "transformers",
        "bert", "gpt", "llm", "large language models", "data analysis", "statistics",
        "data visualization", "feature engineering", "model deployment"
    ],
    # Databases
    "databases": [
        "sql", "mysql", "postgresql", "mongodb", "redis", "cassandra", "oracle",
        "sql server", "sqlite", "dynamodb", "elasticsearch", "neo4j", "firebase",
        "mariadb", "couchdb"
    ],
    # Cloud & DevOps
    "cloud_devops": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "jenkins",
        "gitlab ci", "github actions", "terraform", "ansible", "ci/cd", "devops",
        "linux", "unix", "bash", "shell scripting", "nginx", "apache"
    ],
    # Tools & Frameworks
    "tools": [
        "git", "github", "gitlab", "bitbucket", "jira", "confluence", "postman",
        "swagger", "vs code", "intellij", "eclipse", "visual studio"
    ],
    # Soft Skills
    "soft_skills": [
        "communication", "leadership", "teamwork", "problem solving", 
        "critical thinking", "project management", "agile", "scrum", "kanban",
        "time management", "collaboration", "analytical skills", "presentation"
    ],
    # Other Technical
    "other": [
        "rest api", "graphql", "microservices", "api development", "testing",
        "unit testing", "integration testing", "jest", "pytest", "selenium",
        "websockets", "oauth", "jwt", "security", "blockchain", "iot"
    ]
}

# Flatten skills into single list
ALL_SKILLS = []
for category in SKILL_DATABASE.values():
    ALL_SKILLS.extend(category)

# Add common variations
SKILL_VARIATIONS = {
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "ml": "machine learning",
    "dl": "deep learning",
    "k8s": "kubernetes",
    "tf": "tensorflow",
    "sklearn": "scikit-learn"
}


# ==================== EDUCATION KEYWORDS ====================
EDUCATION_KEYWORDS = [
    "bachelor", "master", "phd", "doctorate", "b.tech", "m.tech", "btech", "mtech",
    "b.e", "m.e", "be", "me", "b.sc", "m.sc", "bsc", "msc", "bca", "mca",
    "mba", "undergraduate", "graduate", "associate", "diploma", "certification",
    "degree", "university", "college", "institute", "school"
]

DEGREE_PATTERNS = [
    r"(?i)bachelor(?:'?s)?(?:\s+of\s+[\w\s]+)?",
    r"(?i)master(?:'?s)?(?:\s+of\s+[\w\s]+)?",
    r"(?i)(?:phd|ph\.d|doctorate)",
    r"(?i)b\.?tech|m\.?tech|b\.?e\.?|m\.?e\.?|b\.?sc\.?|m\.?sc\.?",
    r"(?i)mba|bba|bca|mca"
]


# ==================== REGEX PATTERNS ====================
class RegexPatterns:
    EMAIL = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    PHONE = r'''
        (?:(?:\+|00)?[1-9]{1,3}[\s.-]?)?  # Country code
        (?:\(?\d{2,4}\)?[\s.-]?)?          # Area code
        \d{3,4}[\s.-]?\d{3,4}              # Main number
    '''
    
    GITHUB = r'(?:https?://)?(?:www\.)?github\.com/[\w-]+'
    LINKEDIN = r'(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+'
    
    # Experience patterns (e.g., "Jan 2020 - Dec 2022", "2020-2022")
    DATE_RANGE = r'''
        (?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+)?
        \d{4}\s*[-–]\s*
        (?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+)?
        (?:\d{4}|Present|Current)
    '''


# ==================== ENTITY RULER SETUP ====================
def setup_entity_ruler(nlp):
    """Add entity ruler for better skill and entity detection"""
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
    else:
        ruler = nlp.get_pipe("entity_ruler")
    
    patterns = []
    
    # Add skill patterns
    for skill in ALL_SKILLS:
        patterns.append({"label": "SKILL", "pattern": skill})
        # Add case variations
        patterns.append({"label": "SKILL", "pattern": skill.upper()})
        patterns.append({"label": "SKILL", "pattern": skill.title()})
    
    # Add skill variations
    for abbr, full in SKILL_VARIATIONS.items():
        patterns.append({"label": "SKILL", "pattern": abbr})
    
    # Add education patterns
    for edu in EDUCATION_KEYWORDS:
        patterns.append({"label": "EDUCATION", "pattern": edu})
    
    ruler.add_patterns(patterns)
    return nlp


# Initialize entity ruler
nlp = setup_entity_ruler(nlp)


# ==================== TEXT EXTRACTION ====================
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF with better error handling"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return ""


def extract_text_from_docx(docx_path: str) -> str:
    """Extract text from DOCX files"""
    try:
        doc = docx.Document(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()
    except Exception as e:
        print(f"Error extracting DOCX: {e}")
        return ""


def extract_text_from_file(file_path: str) -> str:
    """Universal text extraction based on file type"""
    path = Path(file_path)
    
    if path.suffix.lower() == '.pdf':
        return extract_text_from_pdf(file_path)
    elif path.suffix.lower() in ['.docx', '.doc']:
        return extract_text_from_docx(file_path)
    elif path.suffix.lower() == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


# ==================== INFORMATION EXTRACTION ====================
def extract_name(text: str, doc) -> Optional[str]:
    """Extract candidate name using NER"""
    # First few lines usually contain name
    first_lines = text.split('\n')[:5]
    first_text = ' '.join(first_lines)
    
    doc_first = nlp(first_text)
    
    for ent in doc_first.ents:
        if ent.label_ == "PERSON":
            return ent.text.strip()
    
    # Fallback: return first line if it looks like a name
    if first_lines and len(first_lines[0].split()) <= 4:
        return first_lines[0].strip()
    
    return None


def extract_emails(text: str) -> List[str]:
    """Extract all email addresses"""
    emails = re.findall(RegexPatterns.EMAIL, text)
    return list(set(emails))  # Remove duplicates


def extract_phone_numbers(text: str) -> List[str]:
    """Extract phone numbers"""
    phones = re.findall(RegexPatterns.PHONE, text, re.VERBOSE)
    # Clean and deduplicate
    cleaned = []
    for phone in phones:
        phone_clean = re.sub(r'[^\d+]', '', phone)
        if 10 <= len(phone_clean) <= 15:  # Valid phone length
            cleaned.append(phone)
    return list(set(cleaned))


def extract_links(text: str) -> Dict[str, List[str]]:
    """Extract social and professional links"""
    return {
        "github": list(set(re.findall(RegexPatterns.GITHUB, text, re.IGNORECASE))),
        "linkedin": list(set(re.findall(RegexPatterns.LINKEDIN, text, re.IGNORECASE)))
    }


def extract_skills_advanced(text: str, doc) -> Dict[str, List[str]]:
    """Advanced skill extraction using NER + pattern matching"""
    text_lower = text.lower()
    
    # Extract using entity ruler
    skills_from_ner = set()
    for ent in doc.ents:
        if ent.label_ == "SKILL":
            skill = ent.text.lower()
            # Map variations to full names
            skill = SKILL_VARIATIONS.get(skill, skill)
            skills_from_ner.add(skill)
    
    # Extract using direct matching
    skills_from_matching = set()
    for skill in ALL_SKILLS:
        # Use word boundaries for better matching
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            skills_from_matching.add(skill)
    
    # Combine both methods
    all_found_skills = skills_from_ner | skills_from_matching
    
    # Categorize skills
    categorized = {category: [] for category in SKILL_DATABASE.keys()}
    for skill in all_found_skills:
        for category, skills_list in SKILL_DATABASE.items():
            if skill in skills_list:
                categorized[category].append(skill)
    
    # Return flattened and categorized
    return {
        "all_skills": sorted(list(all_found_skills)),
        "categorized": {k: sorted(v) for k, v in categorized.items() if v}
    }


# ==================== BACKWARD COMPATIBILITY ====================
def extract_skills_from_text(text: str) -> List[str]:
    """
    Extract skills from text string (legacy function for compatibility)
    
    Args:
        text: Resume text string
    
    Returns:
        List of extracted skills
    """
    if not text or not text.strip():
        return []
    
    try:
        doc = nlp(text)
        result = extract_skills_advanced(text, doc)
        return result["all_skills"]
    except Exception as e:
        print(f"Error extracting skills: {e}")
        return []


def extract_education(text: str) -> List[str]:
    """Extract education information"""
    education = []
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        
        # Check for degree patterns
        for pattern in DEGREE_PATTERNS:
            matches = re.findall(pattern, line, re.IGNORECASE)
            if matches:
                # Get context (current and next lines)
                context = line
                if i + 1 < len(lines):
                    context += " " + lines[i + 1]
                education.append(context.strip())
    
    return list(set(education))[:5]  # Limit to 5 entries


def extract_experience(text: str) -> List[Dict[str, str]]:
    """Extract work experience with date ranges"""
    experiences = []
    date_ranges = re.findall(RegexPatterns.DATE_RANGE, text, re.VERBOSE | re.IGNORECASE)
    
    for date_range in date_ranges:
        experiences.append({
            "period": date_range.strip(),
            "years": calculate_years(date_range)
        })
    
    return experiences


def calculate_years(date_range: str) -> float:
    """Calculate years from date range"""
    try:
        years = re.findall(r'\d{4}', date_range)
        if len(years) >= 2:
            start_year = int(years[0])
            end_year = int(years[1]) if years[1].isdigit() else 2025
            return round(end_year - start_year, 1)
    except:
        pass
    return 0.0


def extract_organizations(doc) -> List[str]:
    """Extract company/organization names"""
    orgs = []
    for ent in doc.ents:
        if ent.label_ == "ORG":
            orgs.append(ent.text)
    return list(set(orgs))


# ==================== MAIN PARSING FUNCTION ====================
def parse_resume(file_path: str) -> Dict:
    """
    Comprehensive resume parser
    
    Args:
        file_path: Path to resume file (PDF, DOCX, or TXT)
    
    Returns:
        Dictionary containing all extracted information
    """
    # Extract text
    text = extract_text_from_file(file_path)
    
    if not text:
        return {"error": "Could not extract text from file"}
    
    # Process with spaCy
    doc = nlp(text)
    
    # Extract all information
    parsed_data = {
        "name": extract_name(text, doc),
        "contact": {
            "emails": extract_emails(text),
            "phones": extract_phone_numbers(text),
            "links": extract_links(text)
        },
        "skills": extract_skills_advanced(text, doc),
        "education": extract_education(text),
        "experience": {
            "positions": extract_experience(text),
            "total_years": sum(exp["years"] for exp in extract_experience(text)),
            "organizations": extract_organizations(doc)
        },
        "summary": {
            "total_skills": len(extract_skills_advanced(text, doc)["all_skills"]),
            "has_email": bool(extract_emails(text)),
            "has_phone": bool(extract_phone_numbers(text)),
            "education_count": len(extract_education(text)),
            "experience_count": len(extract_experience(text))
        }
    }
    
    return parsed_data


# ==================== UTILITY FUNCTIONS ====================
def match_resume_to_job(resume_data: Dict, required_skills: List[str]) -> Dict:
    """
    Calculate match score between resume and job requirements
    
    Args:
        resume_data: Parsed resume data
        required_skills: List of required skills for job
    
    Returns:
        Match analysis with score
    """
    candidate_skills = set(skill.lower() for skill in resume_data["skills"]["all_skills"])
    required_skills_lower = set(skill.lower() for skill in required_skills)
    
    matched_skills = candidate_skills & required_skills_lower
    missing_skills = required_skills_lower - candidate_skills
    
    match_score = (len(matched_skills) / len(required_skills_lower) * 100) if required_skills_lower else 0
    
    return {
        "match_score": round(match_score, 2),
        "matched_skills": sorted(list(matched_skills)),
        "missing_skills": sorted(list(missing_skills)),
        "additional_skills": sorted(list(candidate_skills - required_skills_lower))
    }


def print_resume_summary(parsed_data: Dict):
    """Pretty print resume summary"""
    print("=" * 60)
    print(f"RESUME ANALYSIS: {parsed_data.get('name', 'Unknown')}")
    print("=" * 60)
    
    print(f"\n📧 Contact:")
    print(f"  Emails: {', '.join(parsed_data['contact']['emails']) or 'Not found'}")
    print(f"  Phones: {', '.join(parsed_data['contact']['phones']) or 'Not found'}")
    
    print(f"\n💼 Experience:")
    print(f"  Total Years: {parsed_data['experience']['total_years']}")
    print(f"  Organizations: {', '.join(parsed_data['experience']['organizations'][:3]) or 'Not found'}")
    
    print(f"\n🎓 Education: {parsed_data['summary']['education_count']} entries found")
    
    print(f"\n🛠️  Skills ({parsed_data['summary']['total_skills']} total):")
    for category, skills in parsed_data['skills']['categorized'].items():
        if skills:
            print(f"  {category}: {', '.join(skills[:5])}")
    
    print("=" * 60)


# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    # Example usage
    resume_file = "sample_resume.pdf"  # Change to your file
    
    try:
        # Parse resume
        result = parse_resume(resume_file)
        
        # Print summary
        print_resume_summary(result)
        
        # Example: Match with job requirements
        job_requirements = ["python", "machine learning", "sql", "aws", "docker"]
        match_result = match_resume_to_job(result, job_requirements)
        
        print(f"\n🎯 Job Match Score: {match_result['match_score']}%")
        print(f"   Matched: {', '.join(match_result['matched_skills'])}")
        print(f"   Missing: {', '.join(match_result['missing_skills'])}")
        
    except Exception as e:
        print(f"Error: {e}")
