from collections import defaultdict

def recommend_roles(skills):
    # Normalize skills (lowercase)
    skill_set = {s.strip().lower() for s in skills}
    
    # Define role-to-skills mapping
    role_skills = {
        "Data Scientist": {"python", "machine learning", "data analysis", "pandas", "numpy", "matplotlib", "scikit-learn"},
        "Frontend Developer": {"html", "css", "javascript", "react", "vue", "angular"},
        "Backend Developer": {"python", "flask", "django", "node.js", "api", "fastapi"},
        "Database Administrator": {"sql", "mysql", "postgresql", "mongodb", "database", "oracle"},
        "Software Developer": {"java", "c++", "c#", "python"},
        "AI/ML Engineer": {"deep learning", "neural networks", "tensorflow", "pytorch", "nlp"},
        "DevOps Engineer": {"docker", "kubernetes", "ci/cd", "aws", "azure", "gcp", "linux"},
        "Cybersecurity Analyst": {"cybersecurity", "penetration testing", "network security", "firewall", "encryption"},
    }
    
    # Track match scores
    role_scores = defaultdict(int)
    
    for role, req_skills in role_skills.items():
        # Count matched skills
        matches = skill_set & req_skills
        if matches:
            role_scores[role] = len(matches) / len(req_skills)  # score = % matched
    
    # If no matches, return fallback
    if not role_scores:
        return [{"role": "General Tech Enthusiast", "score": 0.0}]
    
    # Sort roles by score (high → low)
    recommendations = sorted(
        [{"role": role, "score": round(score, 2)} for role, score in role_scores.items()],
        key=lambda x: x["score"],
        reverse=True
    )
    
    return recommendations


# Example usage
skills = ["Python", "Django", "SQL", "Pandas", "Machine Learning"]
print(recommend_roles(skills))
