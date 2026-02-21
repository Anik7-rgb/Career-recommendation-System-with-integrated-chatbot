import json
import os
import requests
from typing import List, Tuple

# --------------------------
# Data (unchanged)
# --------------------------
course_db = {
    "python": ["Python for Everybody – Coursera", "Learn Python – Codecademy"],
    "sql": ["Intro to SQL – Khan Academy", "SQL Basics – DataCamp"],
    "machine learning": ["ML Crash Course – Google", "ML A-Z – Udemy"],
    "html": ["HTML & CSS – FreeCodeCamp"],
    "flask": ["Flask Mega-Tutorial – Miguel Grinberg"]
}

# --------------------------
# Baseline (your original)
# --------------------------
def recommend_courses_baseline(skills: List[str]) -> List[Tuple[str, str]]:
    recommended = []
    for skill, courses in course_db.items():
        if skill not in skills:
            for course in courses:
                recommended.append((skill.title(), course))
    return recommended[:5]  # limit output


# --------------------------
# LLM reranker via LM Studio
# --------------------------
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "mistral-7b-instruct.Q4_K_M.gguf")
LM_TIMEOUT = float(os.getenv("LM_TIMEOUT", "15"))

def _format_candidates(cands: List[Tuple[str, str]]) -> str:
    # Turn candidate tuples into a numbered list the model can parse
    lines = []
    for i, (skill, title) in enumerate(cands, 1):
        lines.append(f"{i}. [{skill}] {title}")
    return "\n".join(lines)

def _llm_chat(messages, temperature=0.2, max_tokens=512):
    url = f"{LM_STUDIO_BASE_URL}/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": LM_STUDIO_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=LM_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    # Standard OpenAI-compatible shape
    return data["choices"][0]["message"]["content"]

def _parse_llm_selection(text: str, num_to_take=5) -> List[int]:
    """
    Parses model output expecting a JSON list of indices or a numbered list.
    Tries JSON first; falls back to extracting leading integers per line.
    Returns 1-based indices.
    """
    # Try to find a JSON array
    text = text.strip()
    try:
        arr = json.loads(text)
        if isinstance(arr, list) and all(isinstance(x, int) for x in arr):
            return arr[:num_to_take]
    except Exception:
        pass

    # Fallback: read lines, take leading ints
    indices = []
    for line in text.splitlines():
        line = line.strip()
        # Common patterns: "1. ...", "1) ...", "1 - ...", or just "1"
        num = ""
        for ch in line:
            if ch.isdigit():
                num += ch
            elif num:
                break
        if num:
            try:
                indices.append(int(num))
            except Exception:
                pass
        if len(indices) >= num_to_take:
            break
    return indices[:num_to_take]

def recommend_courses_llm(skills: List[str], top_k: int = 5) -> List[Tuple[str, str]]:
    """
    1) Use baseline to get a candidate pool (can expand beyond top_k).
    2) Ask local LLM to pick and order the best top_k for the learner.
    3) If LLM unavailable or parsing fails, fall back to baseline.
    """
    # Step 1: create a broader pool (e.g., 15) from baseline
    pool = []
    for skill, courses in course_db.items():
        if skill not in skills:
            for c in courses:
                pool.append((skill.title(), c))
    if not pool:
        return []

    candidate_block = _format_candidates(pool)

    system_msg = (
        "You are a course recommender that picks concise, highly relevant courses for the learner. "
        "Return only the selected item numbers as a JSON array of integers (e.g., [3,1,7,5,2])."
    )
    user_msg = (
        f"Learner's known skills: {', '.join(skills) if skills else 'None'}.\n\n"
        f"Candidates:\n{candidate_block}\n\n"
        f"Task: Choose the best {top_k} items for this learner, optimizing for skill gaps, progression, and quality. "
        f"Return only a JSON array with the chosen item numbers in your recommended order."
    )

    try:
        content = _llm_chat(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=128
        )
        chosen = _parse_llm_selection(content, num_to_take=top_k)
        # Map chosen indices (1-based) back to pool
        ranked = []
        for idx in chosen:
            if 1 <= idx <= len(pool):
                ranked.append(pool[idx - 1])
        if ranked:
            return ranked
    except Exception:
        # Any error -> fallback
        pass

    # Fallback: just return the baseline top_k
    return recommend_courses_baseline(skills)[:top_k]


# --------------------------
# Unified API
# --------------------------
def recommend_courses(skills: List[str], use_llm: bool = True, top_k: int = 5) -> List[Tuple[str, str]]:
    """
    Public entrypoint. Toggle use_llm to enable/disable local reranking.
    """
    if use_llm:
        return recommend_courses_llm(skills, top_k=top_k)
    return recommend_courses_baseline(skills)[:top_k]


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    # Example: learner knows Python and HTML, wants 5 suggestions
    skills = ["python", "html"]
    recs = recommend_courses(skills, use_llm=True, top_k=5)
    for skill, title in recs:
        print(f"{skill}: {title}")
