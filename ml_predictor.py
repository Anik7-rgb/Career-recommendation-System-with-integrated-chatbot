import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model = joblib.load("model.joblib")
encoder = joblib.load("encoder.joblib")
role_names = model.classes_

def predict_top_roles(skills, top_n=5, diversity_threshold=0.6):
    try:
        skills_encoded = encoder.transform([skills])
        probs = model.predict_proba(skills_encoded)[0]
        
        # Get initial top roles based on probability
        initial_top_indices = np.argsort(probs)[::-1][:top_n*2]  # Get more candidates for diversity
        
        # Ensure diversity in recommendations
        diverse_indices = []
        
        # Always include the top role
        diverse_indices.append(initial_top_indices[0])
        
        # Create a simple representation of each role for diversity calculation
        # Using one-hot encoding for role categories
        role_vectors = np.eye(len(role_names))[initial_top_indices]
        
        # Add diverse roles
        for idx in initial_top_indices[1:]:
            # Skip if we already have enough roles
            if len(diverse_indices) >= top_n:
                break
                
            # Check similarity with already selected roles
            is_diverse = True
            for selected_idx in diverse_indices:
                # Simple similarity check - if roles are in completely different categories
                if role_names[idx].split()[0] == role_names[selected_idx].split()[0]:
                    is_diverse = False
                    break
            
            # If this role is diverse enough, add it
            if is_diverse:
                diverse_indices.append(idx)
            
        # If we don't have enough diverse roles, add more from the top roles
        remaining_slots = top_n - len(diverse_indices)
        if remaining_slots > 0:
            for idx in initial_top_indices:
                if idx not in diverse_indices and remaining_slots > 0:
                    diverse_indices.append(idx)
                    remaining_slots -= 1
        
        # Format the results
        top_roles = [(role_names[i].title(), round(probs[i]*100, 2)) for i in diverse_indices[:top_n]]
        return top_roles
    except Exception as e:
        print(f"Error in predict_top_roles: {e}")
        return [("Unknown", 0.0)]
