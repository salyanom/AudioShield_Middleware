from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")


def verify_context(user_request, response):

    user_embedding = model.encode([user_request])

    response_embedding = model.encode([response])

    similarity = cosine_similarity(
        user_embedding,
        response_embedding
    )[0][0]

    return similarity