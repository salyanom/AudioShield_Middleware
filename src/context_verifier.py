import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(
            os.getenv("AUDIOSHIELD_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        )
    return _model


def verify_context(user_request, response):

    model = _get_model()
    user_embedding = model.encode([user_request])

    response_embedding = model.encode([response])

    similarity = cosine_similarity(
        user_embedding,
        response_embedding
    )[0][0]

    return similarity
