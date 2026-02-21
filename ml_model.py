import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

# Expanded Dataset with diverse career roles (skill set -> role)
data = [
    # Tech Roles
    (["python", "machine learning", "numpy", "pandas", "statistics"], "Data Scientist"),
    (["html", "css", "javascript", "react", "ui design"], "Frontend Developer"),
    (["python", "flask", "api", "django", "fastapi"], "Backend Developer"),
    (["java", "c++", "oop", "algorithms", "data structures"], "Software Engineer"),
    (["sql", "database", "mysql", "postgresql", "data modeling"], "Database Administrator"),
    (["html", "python", "sql", "javascript", "api"], "Full Stack Developer"),
    (["cloud", "aws", "azure", "devops", "kubernetes"], "Cloud Architect"),
    (["cybersecurity", "network", "encryption", "security", "penetration testing"], "Security Engineer"),
    
    # Business Roles
    (["marketing", "social media", "content creation", "analytics", "seo"], "Marketing Specialist"),
    (["finance", "accounting", "excel", "financial analysis", "budgeting"], "Financial Analyst"),
    (["sales", "negotiation", "client management", "crm", "presentation"], "Sales Manager"),
    (["project management", "agile", "scrum", "leadership", "planning"], "Project Manager"),
    (["hr", "recruitment", "employee relations", "talent management", "onboarding"], "HR Manager"),
    
    # Creative Roles
    (["design", "photoshop", "illustrator", "typography", "branding"], "Graphic Designer"),
    (["ux", "user research", "wireframing", "prototyping", "usability testing"], "UX Designer"),
    (["writing", "editing", "content strategy", "seo", "storytelling"], "Content Writer"),
    (["video editing", "animation", "after effects", "premiere pro", "storyboarding"], "Video Editor"),
    
    # Healthcare Roles
    (["patient care", "medical knowledge", "clinical", "healthcare", "diagnosis"], "Healthcare Professional"),
    (["data analysis", "healthcare", "medical records", "research", "statistics"], "Healthcare Data Analyst"),
    
    # Education Roles
    (["teaching", "curriculum development", "education", "assessment", "classroom management"], "Educator"),
    (["instructional design", "e-learning", "curriculum", "educational technology", "assessment"], "Instructional Designer")
]

# Convert to DataFrame
df = pd.DataFrame(data, columns=["skills", "role"])

# Binarize skills
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["skills"])
y = df["role"]

# Train model with more estimators for better performance
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Save the model and encoder
joblib.dump(clf, "model.joblib")
joblib.dump(mlb, "encoder.joblib")
