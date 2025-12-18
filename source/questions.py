"""Module to store questionnaire text and options.

This keeps `app.py` clean and makes it easier to edit or test questions.
"""

__all__ = ["questions", "options"]

questions = {
    'q16': "Risk appetite?",
    'q17': "Investment goal?",
    'q18': "Prefer gains or avoid losses?",
    'q19': "If your portfolio fell 20% quickly, you would:",
    'q13': "Investable funds?",
    'q6': "Investment knowledge level?",
    'q7': "Investment experience?",
    'q8': "Trading frequency?"
}

options = {
    'q16': {'a': "Very high", 'b': "High", 'c': "Moderate", 'd': "Low", 'e': "Very low"},
    'q17': {'a': "High returns", 'b': "Accept some loss", 'c': "Steady income", 'd': "Stable income", 'e': "Capital preservation"},
    'q18': {'a': "Prefer gains", 'b': "Often prefer gains", 'c': "Both equally", 'd': "Often avoid losses", 'e': "Always avoid losses"},
    'q19': {'a': "Buy more", 'b': "Rebalance (buy)", 'c': "Hold", 'd': "Sell some", 'e': "Sell all"},
    'q13': {'a': ">1M €", 'b': "300k–1M €", 'c': "80k–300k €", 'd': "30k–80k €", 'e': "≤30k €"},
    'q6': {'a': "Low", 'b': "Average", 'c': "Good", 'd': "Expert"},
    'q7': {'a': "None", 'b': "Moderate", 'c': "Experienced", 'd': "Very experienced"},
    'q8': {'a': "Rare (1–2/yr)", 'b': "Occasional (2–3/mo)", 'c': "Monthly/biweekly", 'd': "Very often (≥2/wk)"}
}
