import numpy as np
import pandas as pd  # Import pandas
from transformers import pipeline
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity  # Add this import for cosine similarity
import matplotlib.pyplot as plt

# Simulated dataset
data = {
    "item_id": [1, 2, 3, 4],
    "title": [
        "Introduction to AI",
        "Machine Learning Basics",
        "Deep Learning Explained",
        "Neural Networks for Beginners"
    ],
    "image_path": [
        "images/ai.jpg",
        "images/ml.jpg",
        "images/dl.jpg",
        "images/nn.jpg"
    ],
    # Precomputed image embeddings (e.g., from CLIP)
    "image_embedding": [
        np.array([0.1, 0.2, 0.3]),
        np.array([0.4, 0.5, 0.6]),
        np.array([0.7, 0.8, 0.9]),
        np.array([0.2, 0.3, 0.4])
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Load pre-trained NLP model for text embeddings
text_embedder = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")

def get_text_embeddings(text):
    """Generate text embeddings using a pre-trained model."""
    embeddings = text_embedder(text)[0]  # Get embeddings for the first sentence
    return np.mean(embeddings, axis=0)  # Average pooling

# Add text embeddings to the DataFrame
df["text_embedding"] = df["title"].apply(get_text_embeddings)

def combine_multimodal_features(row):
    """Combine text and image embeddings."""
    text_embedding = np.array(row["text_embedding"])
    image_embedding = np.array(row["image_embedding"])
    # Make sure both arrays are 1D for concatenation
    return np.concatenate([text_embedding, image_embedding])

# Add combined embeddings to the DataFrame
df["combined_embedding"] = df.apply(combine_multimodal_features, axis=1)

def recommend_items(user_query, top_n=2):
    """Recommend items based on user query."""
    # Generate user query embedding
    user_text_embedding = get_text_embeddings(user_query)
    user_image_embedding = np.array([0.5, 0.5, 0.5])  # Simulated user image preference
    user_combined_embedding = np.concatenate([user_text_embedding, user_image_embedding])
    
    # Compute similarity scores
    similarities = []
    for _, row in df.iterrows():
        combined_emb = np.array(row["combined_embedding"])
        sim = cosine_similarity([user_combined_embedding], [combined_emb])[0][0]
        similarities.append(sim)
    
    df_copy = df.copy()
    df_copy["similarity"] = similarities
    
    # Sort by similarity and return top recommendations
    recommendations = df_copy.sort_values(by="similarity", ascending=False).head(top_n)
    return recommendations[["item_id", "title", "similarity"]]

def generate_explanation(recommended_item, user_query):
    """Generate a personalized explanation for the recommendation."""
    title = recommended_item["title"]
    similarity_score = recommended_item["similarity"]
    explanation = (
        f"This item ('{title}') was recommended because it matches your interest in '{user_query}'. "
        f"The similarity score is {similarity_score:.2f}, indicating a strong alignment with your preferences."
    )
    return explanation

# User query
user_query = "I want to learn about deep learning"

# Get recommendations
recommendations = recommend_items(user_query, top_n=2)
print("Recommendations:")
print(recommendations)

# Generate explanations
for _, row in recommendations.iterrows():
    explanation = generate_explanation(row, user_query)
    print("\nExplanation:")
    print(explanation)