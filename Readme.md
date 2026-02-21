# SkillSense: AI-Powered Career Recommendation System 💼✨

SkillSense is a web-based intelligent system that:
- Parses a user’s resume (PDF),
- Extracts technical skills using NLP,
- Predicts top career roles using ML,
- Suggests relevant jobs and courses,
- Uses a local AI model (via LM Studio) to answer custom career queries.

---

## 🚀 Features

- 📄 Resume Skill Extraction (PDF parser + NLP)
- 🤖 ML-based Career Role Prediction
- 🌐 Real-time Job Scraping by Role
- 📚 Smart Course Suggestions
- 🧠 AI Assistant powered by Local LLM (Mistral via LM Studio)
- 🌐 Fully Offline AI Chat Option (No OpenAI API Key required!)

---

## 🧰 Tech Stack

| Area | Tech |
|------|------|
| Backend | Python, Flask |
| AI/ML | Scikit-learn, Pandas, NumPy |
| Resume Parsing | PyMuPDF (fitz), Regex |
| Job Scraping | Requests, BeautifulSoup |
| AI Assistant | LM Studio (OpenAI-compatible API server) |
| Frontend | HTML, CSS (Bootstrap), Jinja2 Templates |
| Model | Mistral-7B-Instruct via LM Studio |

---

## 🛠️ Setup Instructions

### 1. 📂 Clone the Repository

```bash
git clone https://github.com/yourusername/skillsense.git
cd skillsense
