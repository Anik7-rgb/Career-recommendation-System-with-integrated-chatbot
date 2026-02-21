import logging
import sys
import os
import json
import requests # type: ignore
import time
from ml_predictor import predict_top_roles
from job_scraper import get_jobs_for_role
from resume_parser import extract_skills_from_text, extract_text_from_pdf
from recommender import recommend_roles
from flask import Flask, request, render_template, jsonify
print("✅ All modules imported successfully")

# Configure minimal logging for better performance
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# Import with error handling
try:
    # from ml_predictor import predict_top_roles
    # from job_scraper import get_jobs_for_role
    # from resume_parser import extract_skills_from_text, extract_text_from_pdf
    # from recommender import recommend_roles
    # from flask import Flask, request, render_template, jsonify
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --------------------------
# OPTIMIZED LM Studio Configuration for SPEED
# --------------------------
LM_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")

# SPEED OPTIMIZED: Use smaller, faster models
CHATBOT_MODEL = os.getenv("CHATBOT_MODEL", "phi-3-mini-4k-instruct")  # Changed to smaller model
CHATBOT_TIMEOUT = float(os.getenv("CHATBOT_TIMEOUT", "20"))  # Reduced timeout

COURSE_MODEL = os.getenv("COURSE_MODEL", "phi-3-mini-4k-instruct")  # Same fast model
COURSE_TIMEOUT = float(os.getenv("COURSE_TIMEOUT", "15"))  # Reduced timeout

# Backward compatibility
LM_MODEL = os.getenv("LM_STUDIO_MODEL", "phi-3-mini-4k-instruct")
LM_TIMEOUT = float(os.getenv("LM_TIMEOUT", "10"))

# --------------------------
# Course database - PRESERVED
# --------------------------
course_db = {
    "python": ["Python for Everybody – Coursera", "Learn Python – Codecademy"],
    "sql": ["Intro to SQL – Khan Academy", "SQL Basics – DataCamp"],
    "machine learning": ["ML Crash Course – Google", "ML A-Z – Udemy"],
    "html": ["HTML & CSS – FreeCodeCamp"],
    "flask": ["Flask Mega-Tutorial – Miguel Grinberg"],
    "javascript": ["JavaScript Basics – MDN", "JS Complete Course – Udemy"],
    "react": ["React Tutorial – Official Docs", "React Masterclass – Scrimba"],
    "node.js": ["Node.js Tutorial – W3Schools", "Node.js Complete Guide – Udemy"],
    "data analysis": ["Data Analysis with Python – FreeCodeCamp", "Pandas Tutorial – Kaggle"],
    "aws": ["AWS Cloud Practitioner – AWS", "AWS Solutions Architect – A Cloud Guru"]
}

def recommend_courses_baseline(skills):
    recommended = []
    for skill, courses in course_db.items():
        if skill not in [s.lower() for s in skills]:
            for course in courses:
                recommended.append((skill.title(), course))
    return recommended[:5]

# --------------------------
# SPEED OPTIMIZED LLM Functions
# --------------------------
def _format_candidates(cands):
    lines = []
    for i, (skill, title) in enumerate(cands, 1):
        lines.append(f"{i}. [{skill}] {title}")
    return "\n".join(lines)

def _llm_chat_fast(messages, model_name, timeout, temperature=0.3, max_tokens=150):
    """OPTIMIZED: Single attempt, reduced tokens, faster temperature"""
    url = f"{LM_BASE_URL}/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        # SPEED OPTIMIZATIONS
        "top_k": 20,      # Reduced from 40
        "top_p": 0.8,     # Reduced from 0.95
        "repeat_penalty": 1.2
    }
    
    # Single attempt only for speed
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

def _parse_llm_selection_fast(text, num_to_take=5):
    """OPTIMIZED: Faster parsing"""
    indices = []
    for line in text.splitlines()[:10]:  # Only check first 10 lines
        for char in line:
            if char.isdigit():
                try:
                    num = int(char)
                    if 1 <= num <= 20 and num not in indices:
                        indices.append(num)
                        if len(indices) >= num_to_take:
                            return indices
                except:
                    continue
    return indices[:num_to_take]

# --------------------------
# SPEED OPTIMIZED Course Recommender
# --------------------------
def recommend_courses_llm(skills, top_k=5):
    # Quick fallback for empty skills
    if not skills:
        return recommend_courses_baseline([])
    
    pool = []
    known = {s.lower() for s in skills}
    for skill, courses in course_db.items():
        if skill not in known:
            for c in courses:
                pool.append((skill.title(), c))
    
    if not pool or len(pool) < 5:
        return recommend_courses_baseline(skills)

    candidate_block = _format_candidates(pool[:15])  # Limit candidates for speed
    
    # SIMPLIFIED system message for faster processing
    system_msg = "Return only a JSON array like [1,2,3,4,5] for the best course numbers."
    user_msg = f"Skills: {', '.join(skills[:5])}\n\nCourses:\n{candidate_block}\n\nPick {top_k} best: "

    try:
        content = _llm_chat_fast(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            model_name=COURSE_MODEL,
            timeout=COURSE_TIMEOUT,
            temperature=0.1,  # Lower for faster, more predictable output
            max_tokens=50     # Very limited for speed
        )
        
        chosen = _parse_llm_selection_fast(content, num_to_take=top_k)
        ranked = []
        for idx in chosen:
            if 1 <= idx <= len(pool):
                ranked.append(pool[idx - 1])
        
        return ranked if ranked else recommend_courses_baseline(skills)[:top_k]
        
    except Exception as e:
        print(f"LLM course rec failed: {e}, using fallback")
        return recommend_courses_baseline(skills)[:top_k]

def recommend_courses(skills, use_llm=True, top_k=5):
    if use_llm:
        return recommend_courses_llm(skills, top_k=top_k)
    return recommend_courses_baseline(skills)[:top_k]

# --------------------------
# SPEED OPTIMIZED AI Query
# --------------------------
def ai_answer_query(query, extracted_skills):
    # Limit input size for speed
    query = query[:200] if len(query) > 200 else query
    skills_str = ', '.join(extracted_skills[:5])  # Limit skills
    
    # SIMPLIFIED prompt for faster processing
    prompt = f"Skills: {skills_str}. Query: {query}. Brief career advice (2 sentences max):"

    try:
        response = requests.post(
            f"{LM_BASE_URL}/chat/completions",
            json={
                "model": CHATBOT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5,
                "max_tokens": 100,  # Reduced for speed
                "top_k": 20,
                "top_p": 0.8
            },
            timeout=CHATBOT_TIMEOUT
        )
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        return "Error: Please try a shorter question or check if LM Studio is running."

# --------------------------
# SPEED OPTIMIZED Chatbot
# --------------------------
def get_lm_studio_response(user_message):
    """SUPER FAST chatbot optimized for speed"""
    try:
        # Limit message length for speed
        if len(user_message) > 500:
            user_message = user_message[:500]
        
        # MINIMAL system prompt for speed
        system_prompt = "You are a helpful career assistant. Give brief, practical advice in 2-3 sentences."
        
        payload = {
            "model": CHATBOT_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.6,
            "max_tokens": 150,  # Much smaller for speed
            "stream": False,
            # SPEED SETTINGS
            "top_k": 15,
            "top_p": 0.8,
            "repeat_penalty": 1.1
        }
        
        # Single fast request - no retries
        response = requests.post(
            f"{LM_BASE_URL}/chat/completions", 
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=CHATBOT_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return "I'm processing your request. Please ensure LM Studio is running with a fast model like Phi-3."
            
    except requests.exceptions.Timeout:
        return "Response taking too long. Try a shorter question or switch to a faster model in LM Studio."
    except requests.exceptions.ConnectionError:
        return "Can't connect to LM Studio. Please ensure it's running on localhost:1234."
    except Exception as e:
        return "Error processing request. Please try again with a simpler question."

# --------------------------
# ROUTES - All preserved with speed optimizations
# --------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """PRESERVED with speed optimizations"""
    try:
        file = request.files['resume_file']
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        custom_query = request.form.get('custom_query', '').strip()

        resume_text = extract_text_from_pdf(filepath)
        extracted_skills = extract_skills_from_text(resume_text)

        top_roles = predict_top_roles(extracted_skills)
        
        if custom_query and any(word in custom_query.lower() for word in ["role", "job", "career"]):
            job_query = custom_query[:50]  # Limit length
        else:
            job_query = top_roles[0][0] if top_roles else "Software Developer"

        jobs = get_jobs_for_role(job_query)
        courses = recommend_courses(extracted_skills, use_llm=True, top_k=5)

        ai_response = ""
        if custom_query:
            ai_response = ai_answer_query(custom_query, extracted_skills)

        return render_template(
            'index.html',
            filename=filename,
            skills=extracted_skills,
            top_roles=top_roles,
            jobs=jobs,
            courses=courses,
            custom_query=custom_query,
            job_query=job_query,
            ai_response=ai_response
        )
    
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """SPEED OPTIMIZED chatbot endpoint"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Please enter a message'}), 400
        
        if len(user_message) > 500:
            return jsonify({'response': 'Please ask a shorter question for faster responses.'}), 200
        
        start_time = time.time()
        response_text = get_lm_studio_response(user_message)
        processing_time = time.time() - start_time
        
        print(f"⚡ Response in {processing_time:.2f}s")
        
        return jsonify({'response': response_text})
        
    except Exception as e:
        return jsonify({'response': 'Error: Please try a simpler question.'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Server error'}), 500

# --------------------------
# OPTIMIZED STARTUP
# --------------------------
if __name__ == '__main__':
    print("🚀 Starting SPEED OPTIMIZED SkillSense...")
    print("⚡ Fast models configured")
    print("⚡ Reduced timeouts and token limits")
    print("⚡ Single-attempt requests for speed")
    
    app.run(
        debug=False,  # Disabled for better performance
        use_reloader=False,
        host='127.0.0.1',
        port=5000,
        threaded=True
    )
