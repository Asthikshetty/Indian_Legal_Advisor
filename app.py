"""
Indian Legal Assistant - A comprehensive legal assistance tool with web scraping and AI analysis

This implementation combines:
1. Web scraping of Indian legal sources (Indian Kanoon, Legislative.gov.in, Supreme Court)
2. Legal text analysis using AI models
3. User-friendly Flask interface for accessing legal advice
"""

# app.py - Main Flask Application
from flask import Flask, render_template, request, jsonify, redirect, url_for
import requests
import json
import time
import re
import os
from bs4 import BeautifulSoup
from urllib.parse import urlencode, urlparse
from werkzeug.utils import secure_filename
import PyPDF2
import pytesseract
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
from transformers import pipeline
import logging
import redis
from functools import wraps

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key')

# Initialize Redis for caching (if available)
try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=0,
        decode_responses=True
    )
    # Test connection
    redis_client.ping()
    REDIS_AVAILABLE = True
    logger.info("Redis cache connected successfully")
except:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, caching disabled")

# Initialize AI models
try:
    # Google Gemini setup
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini = genai.GenerativeModel('gemini-pro')
        AI_MODEL_AVAILABLE = True
        logger.info("Google Gemini AI model initialized")
    else:
        AI_MODEL_AVAILABLE = False
        logger.warning("Gemini API key not found, AI analysis will be limited")
        
    # Legal BERT classifier (optional)
    try:
        legal_classifier = pipeline(
            "text-classification", 
            model="nlpaueb/legal-bert-base-uncased",
            tokenizer="nlpaueb/legal-bert-base-uncased"
        )
        logger.info("Legal BERT classifier loaded")
        LEGAL_CLASSIFIER_AVAILABLE = True
    except Exception as e:
        logger.warning(f"Legal BERT classifier not loaded: {str(e)}")
        LEGAL_CLASSIFIER_AVAILABLE = False
        
except Exception as e:
    AI_MODEL_AVAILABLE = False
    LEGAL_CLASSIFIER_AVAILABLE = False
    logger.error(f"Error initializing AI models: {str(e)}")

# Legal websites configuration
LEGAL_SOURCES = {
    'indiankanoon': {
        'base_url': 'https://indiankanoon.org/search/',
        'params': {'formInput': '', 'pagenum': 1}
    },
    'legislative': {
        'base_url': 'https://legislative.gov.in/constitution-of-india/'
    },
    'supremecourt': {
        'base_url': 'https://main.sci.gov.in/judgments'
    }
}

# Legal Categories and Relevant Sections
LEGAL_CATEGORIES = {
    'RENT_DISPUTE': [
        "Transfer of Property Act Section 106",
        "Delhi Rent Control Act Section 6",
        "Maharashtra Rent Control Act Section 7"
    ],
    'PROPERTY_DISPUTE': [
        "Transfer of Property Act Section 54",
        "Registration Act Section 17",
        "Specific Relief Act Section 10"
    ],
    'CONTRACT_BREACH': [
        "Indian Contract Act Section 73",
        "Indian Contract Act Section 74",
        "Specific Relief Act Section 14"
    ],
    'FAMILY_DISPUTE': [
        "Hindu Marriage Act Section 13",
        "Special Marriage Act Section 27",
        "Hindu Succession Act Section 6"
    ],
    'CRIMINAL_OFFENSE': [
        "Indian Penal Code Section 319",
        "Code of Criminal Procedure Section 154",
        "Evidence Act Section 25"
    ]
}

# Helper Functions
def allowed_file(filename):
    """Check if uploaded file type is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def rate_limited(max_per_minute):
    """Rate limiting decorator to prevent excessive scraping"""
    def decorator(func):
        last_called = {}
        min_interval = 60.0 / max_per_minute
        
        @wraps(func)
        def wrapper(url, *args, **kwargs):
            current_time = time.time()
            domain = urlparse(url).netloc
            
            if domain in last_called:
                elapsed = current_time - last_called[domain]
                if elapsed < min_interval:
                    wait_time = min_interval - elapsed
                    logger.info(f"Rate limiting: Waiting {wait_time:.2f}s for {domain}")
                    time.sleep(wait_time)
            
            last_called[domain] = time.time()
            return func(url, *args, **kwargs)
        return wrapper
    return decorator

@rate_limited(10)  # Max 10 requests per minute per domain
def make_request(url, method='get', params=None, headers=None):
    """Make HTTP request with rate limiting and proper headers"""
    if headers is None:
        headers = {
            'User-Agent': 'LegalAssistant/1.0 (Educational Project)',
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Language': 'en-US,en;q=0.5'
        }
    
    try:
        if method.lower() == 'get':
            response = requests.get(url, params=params, headers=headers, timeout=30)
        else:
            response = requests.post(url, data=params, headers=headers, timeout=30)
        
        # Log response code
        logger.info(f"Request to {url}: Status {response.status_code}")
        
        # Check if rate limited or blocked
        if response.status_code == 429:
            logger.warning(f"Rate limited by {url}")
            time.sleep(60)  # Wait longer if rate limited
            return None
        
        return response if response.status_code == 200 else None
    except Exception as e:
        logger.error(f"Error requesting {url}: {str(e)}")
        return None

def classify_legal_issue(text):
    """Classify the legal issue category"""
    if not text or len(text.strip()) < 10:
        return "UNKNOWN"
        
    # Use NLP model if available
    if LEGAL_CLASSIFIER_AVAILABLE:
        try:
            result = legal_classifier(text)[0]
            return result['label']
        except Exception as e:
            logger.error(f"Error classifying text: {str(e)}")
    
    # Fallback to keyword matching
    keywords = {
        'RENT_DISPUTE': ['rent', 'tenant', 'landlord', 'lease', 'eviction'],
        'PROPERTY_DISPUTE': ['property', 'sale deed', 'ownership', 'title', 'possession'],
        'CONTRACT_BREACH': ['contract', 'agreement', 'breach', 'terms', 'damages'],
        'FAMILY_DISPUTE': ['divorce', 'maintenance', 'custody', 'marriage', 'inheritance'],
        'CRIMINAL_OFFENSE': ['criminal', 'complaint', 'police', 'fir', 'offense', 'theft']
    }
    
    text_lower = text.lower()
    scores = {}
    
    for category, category_keywords in keywords.items():
        score = sum(1 for keyword in category_keywords if keyword.lower() in text_lower)
        scores[category] = score
    
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    return "UNKNOWN"

def generate_search_keywords(text):
    """Generate relevant search keywords from input text"""
    # Extract keywords based on legal domain
    keywords = []
    
    # Extract names of acts that might be mentioned
    act_pattern = r'([A-Z][a-z]+ (?:Act|Code|Rules|Bill)(?:\s+of\s+\d{4})?)'
    acts = re.findall(act_pattern, text)
    if acts:
        keywords.extend(acts)
    
    # Extract section numbers
    section_pattern = r'[Ss]ection\s+(\d+\w*)'
    sections = re.findall(section_pattern, text)
    if sections:
        keywords.extend([f"section {s}" for s in sections])
    
    # Add key phrases from issue classification
    issue_type = classify_legal_issue(text)
    if issue_type in LEGAL_CATEGORIES:
        # Add relevant legal sections as keywords
        relevant_sections = [s.split(' Section')[0] for s in LEGAL_CATEGORIES[issue_type][:2]]
        keywords.extend(relevant_sections)
    
    # Add additional context keywords
    context_keywords = []
    if 'rent' in text.lower() or 'tenant' in text.lower():
        context_keywords.append('rent control')
    if 'property' in text.lower() or 'sale' in text.lower():
        context_keywords.append('property transfer')
    if 'divorce' in text.lower() or 'marriage' in text.lower():
        context_keywords.append('marriage act')
    
    keywords.extend(context_keywords)
    
    # Ensure we have enough keywords by adding common words from input
    if len(keywords) < 3:
        common_legal_terms = ['legal', 'rights', 'court', 'law', 'judge', 'case']
        additional_words = [word for word in text.split() 
                          if len(word) > 3 and word.lower() not in common_legal_terms]
        keywords.extend(additional_words[:3])
    
    # Deduplicate and limit
    return list(dict.fromkeys(keywords))[:5]

def extract_text_from_file(file_path):
    """Extract text from uploaded files (PDF, TXT, images)"""
    try:
        file_ext = file_path.rsplit('.', 1)[1].lower()
        
        if file_ext == 'pdf':
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = '\n'.join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif file_ext == 'txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        elif file_ext in ['png', 'jpg', 'jpeg']:
            text = pytesseract.image_to_string(Image.open(file_path))
        else:
            text = ""
            
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {str(e)}")
        return ""

def scrape_indian_laws(query):
    """Scrape Indian legal websites for relevant information"""
    legal_data = []
    
    # Check cache first
    cache_key = f"legal_search:{hash(query)}"
    if REDIS_AVAILABLE:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            logger.info(f"Cache hit for query: {query[:30]}...")
            return json.loads(cached_data)
    
    # Generate search keywords
    keywords = generate_search_keywords(query)
    search_term = " ".join(keywords)
    logger.info(f"Generated search keywords: {keywords}")
    
    try:
        # 1. Scrape Indian Kanoon
        ik_params = LEGAL_SOURCES['indiankanoon']['params'].copy()
        ik_params['formInput'] = search_term
        
        response = make_request(
            LEGAL_SOURCES['indiankanoon']['base_url'],
            params=ik_params
        )
        
        if response:
            soup = BeautifulSoup(response.text, 'lxml')
            results = soup.select('.result')[:5]  # Get top 5 results
            
            for result in results:
                try:
                    title_elem = result.select_one('.result_title a')
                    if not title_elem:
                        continue
                        
                    title = title_elem.text.strip()
                    link = "https://indiankanoon.org" + title_elem['href'] if title_elem['href'].startswith('/doc') else title_elem['href']
                    
                    citation_elem = result.select_one('.docsource')
                    citation = citation_elem.text.strip() if citation_elem else "Citation not available"
                    
                    sections = [s.text.strip() for s in result.select('.doc_cite')]
                    
                    # Get a short snippet
                    snippet_elem = result.select_one('.snippet')
                    snippet = snippet_elem.text.strip() if snippet_elem else ""
                    
                    case = {
                        'source': 'Indian Kanoon',
                        'title': title,
                        'citation': citation,
                        'sections': sections,
                        'snippet': snippet[:200] + '...' if len(snippet) > 200 else snippet,
                        'link': link
                    }
                    legal_data.append(case)
                except Exception as e:
                    logger.error(f"Error parsing IndianKanoon result: {str(e)}")
                    continue

        # 2. Scrape Legislative.gov.in for relevant acts
        issue_type = classify_legal_issue(query)
        if issue_type in LEGAL_CATEGORIES:
            act_names = set()
            for section in LEGAL_CATEGORIES[issue_type]:
                act_name = section.split(' Section')[0].strip()
                act_names.add(act_name)
            
            for act_name in list(act_names)[:2]:  # Top 2 relevant acts
                act_search_url = f"https://legislative.gov.in/search/{act_name.replace(' ', '+')}"
                response = make_request(act_search_url)
                
                if response:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    results = soup.select('.view-content .views-row')[:2]
                    
                    for result in results:
                        try:
                            title_elem = result.select_one('h2 a')
                            if not title_elem:
                                continue
                                
                            title = title_elem.text.strip()
                            link = title_elem['href']
                            if not link.startswith('http'):
                                link = "https://legislative.gov.in" + link
                            
                            legal_data.append({
                                'source': 'Legislative.gov.in',
                                'title': title,
                                'type': 'statute',
                                'link': link
                            })
                        except Exception as e:
                            logger.error(f"Error parsing legislative result: {str(e)}")
                            continue

        # 3. Scrape Supreme Court website (if relevant)
        if issue_type in ['PROPERTY_DISPUTE', 'CRIMINAL_OFFENSE', 'CONTRACT_BREACH']:
            # This is more complex as SC site might use forms/JS
            # For this implementation, we'll include a placeholder
            legal_data.append({
                'source': 'Supreme Court of India',
                'title': f"Latest judgments related to {issue_type.replace('_', ' ').title()}",
                'type': 'judgment',
                'link': 'https://main.sci.gov.in/judgments'
            })
    
    except Exception as e:
        logger.error(f"Scraping error: {str(e)}")
    
    # Cache results
    if REDIS_AVAILABLE and legal_data:
        redis_client.setex(
            cache_key,
            3600,  # Cache for 1 hour
            json.dumps(legal_data)
        )
    
    return legal_data

def generate_legal_advice(query, legal_data):
    """Generate legal advice using AI"""
    if not AI_MODEL_AVAILABLE:
        return {
            'summary': "AI analysis not available. Please refer to the cases and laws listed below.",
            'issue_type': classify_legal_issue(query),
            'applicable_laws': LEGAL_CATEGORIES.get(classify_legal_issue(query), []),
            'recommendation': "Please consult with a qualified legal professional for specific advice."
        }
    
    try:
        # Prepare context from scraped legal data
        legal_context = ""
        for i, item in enumerate(legal_data[:3]):  # Use top 3 results for context
            if 'title' in item:
                legal_context += f"Legal Source {i+1}: {item['title']}\n"
            if 'sections' in item and item['sections']:
                legal_context += f"Sections: {', '.join(item['sections'])}\n"
            if 'snippet' in item and item['snippet']:
                legal_context += f"Details: {item['snippet']}\n\n"
        
        # Get issue type and applicable laws
        issue_type = classify_legal_issue(query)
        applicable_laws = LEGAL_CATEGORIES.get(issue_type, [])
        
        # Generate prompt for Gemini
        prompt = f"""
        As a legal assistant specializing in Indian law, provide a concise analysis of the following legal issue:
        
        User Query: {query}
        
        Issue Classification: {issue_type.replace('_', ' ').title()}
        
        Applicable Laws:
        {' '.join(applicable_laws)}
        
        Legal Context from Indian Legal Sources:
        {legal_context}
        
        Provide a 3-part response:
        1. A brief summary of the legal issue (2-3 sentences)
        2. The key applicable laws and their relevance (2-3 sentences)
        3. Initial steps the person should consider (3-4 bullet points)
        
        Keep your advice practical, accurate, and ethical. Include a disclaimer that this is general information and not specific legal advice.
        """
        
        # Call Gemini API
        response = gemini.generate_content(prompt)
        response_text = response.text
        
        # Parse the response
        sections = response_text.split('\n\n')
        
        summary = ""
        laws_text = ""
        recommendations = []
        
        for section in sections:
            if "summary" in section.lower() or section.strip().startswith("1."):
                summary = section.split(":", 1)[1].strip() if ":" in section else section.replace("1.", "").strip()
            elif "applicable laws" in section.lower() or section.strip().startswith("2."):
                laws_text = section.split(":", 1)[1].strip() if ":" in section else section.replace("2.", "").strip()
            elif "steps" in section.lower() or "consider" in section.lower() or section.strip().startswith("3."):
                # Extract bullet points
                for line in section.split("\n"):
                    if line.strip().startswith("-") or line.strip().startswith("•") or line.strip().startswith("*"):
                        recommendations.append(line.strip()[2:].strip())
                    elif line.strip().startswith("3."):
                        recommendations.append(line.replace("3.", "").strip())
        
        # Ensure we have recommendations
        if not recommendations:
            recommendations = ["Consult with a legal professional", 
                              "Gather all relevant documents", 
                              "Consider alternative dispute resolution methods"]
        
        return {
            'summary': summary,
            'issue_type': issue_type.replace('_', ' ').title(),
            'applicable_laws': applicable_laws,
            'laws_analysis': laws_text,
            'recommendations': recommendations,
            'disclaimer': "This is general information only, not specific legal advice. Please consult with a qualified legal professional."
        }
    except Exception as e:
        logger.error(f"Error generating legal advice: {str(e)}")
        return {
            'summary': "Unable to generate detailed analysis at this time.",
            'issue_type': classify_legal_issue(query),
            'applicable_laws': LEGAL_CATEGORIES.get(classify_legal_issue(query), []),
            'recommendations': ["Consult with a qualified legal professional", 
                              "Gather all relevant documents", 
                              "Research the legal provisions mentioned"],
            'disclaimer': "This is general information only, not specific legal advice."
        }

# Flask Routes
@app.route('/')
def home():
    # Only render template without passing analysis
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Process user query and provide legal analysis"""
    try:
        # Get input data
        text_input = request.form.get('text', '')
        uploaded_file = request.files.get('file')
        file_text = ''

        # Process uploaded file if any
        if uploaded_file and allowed_file(uploaded_file.filename):
            filename = secure_filename(uploaded_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Ensure upload directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Save and process file
            uploaded_file.save(filepath)
            file_text = extract_text_from_file(filepath)

        # Combine inputs
        full_text = f"{text_input}\n\n{file_text}".strip()
        
        if not full_text:
            return render_template('result.html', 
                error="Please provide text input or upload a document")

        # Log the query (without PII)
        query_len = len(full_text)
        logger.info(f"Processing query of length {query_len} characters")
        
        # Scrape legal information
        legal_data = scrape_indian_laws(full_text)
        
        # Generate legal analysis
        analysis = generate_legal_advice(full_text, legal_data)
        analysis['cases'] = legal_data

        return render_template('result.html', analysis=analysis)

    except Exception as e:
        logger.error(f"Error in analyze route: {str(e)}")
        return render_template('result.html', error=f"An error occurred: {str(e)}")

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for legal analysis"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query parameter'}), 400
            
        query = data['query']
        legal_data = scrape_indian_laws(query)
        analysis = generate_legal_advice(query, legal_data)
        
        return jsonify({
            'analysis': analysis,
            'legal_sources': legal_data
        })
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Check if Tesseract is installed
    try:
        pytesseract.get_tesseract_version()
        logger.info("Tesseract OCR available")
    except:
        logger.warning("Tesseract OCR not found. Image processing will be limited.")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_DEBUG', 'False') == 'True')