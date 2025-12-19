"""Investment Recommendation System - Source Package."""

from .data_loader import load_data, preprocess_data, build_rating_matrix
from .recommender import hybrid_recommendation, matrix_factorization
from .evaluation import compute_roi_at_k, compute_ndcg_at_k
from .profile_manager import process_questionnaire_responses, update_customer_profile
from .styles import apply_minimal_css, html
from .initialization import get_data_and_models

__all__ = [
    'load_data',
    'preprocess_data', 
    'build_rating_matrix',
    'hybrid_recommendation',
    'matrix_factorization',
    'compute_roi_at_k',
    'compute_ndcg_at_k',
    'process_questionnaire_responses',
    'update_customer_profile',
    'apply_minimal_css',
    'html',
    'get_data_and_models',
]