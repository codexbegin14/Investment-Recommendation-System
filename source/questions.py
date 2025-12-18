"""Module to store questionnaire text and options.

This keeps `app.py` clean and makes it easier to edit or test questions.
"""

__all__ = ["questions", "options"]

questions = {
    'q16': "How would you rate your appetite for 'risk'?",
    'q17': "Which of the following sentences best fits your investment expectations?",
    'q18': "In the event that you have to make a financial decision, are you more concerned with potential losses or potential gains?",
    'q19': "Assuming that the value of your investment declines by 20% in short period of time, then your risk tolerance would be:",
    'q13': "What is the amount of funds you have invested or have available to invest?",
    'q6': "How would you describe your level of investment knowledge?",
    'q7': "What is your investment experience?",
    'q8': "How often on average did you make trades in various financial instruments in the last three years?"
}

options = {
    'q16': {'a': "Particularly high...", 'b': "Probably high...", 'c': "Moderate...", 'd': "Low...", 'e': "Too low..."},
    'q17': {'a': "I am willing to take more risk...", 'b': "I can accept reductions...", 'c': "I desire steady income...", 'd': "I wish to achieve a stable income...", 'e': "I wish to maintain the value..."},
    'q18': {'a': "Always the potential profits", 'b': "Usually the potential profits", 'c': "Both potential gains and potential losses", 'd': "Usually the potential losses", 'e': "Always potential losses"},
    'q19': {'a': "I would see this as an opportunity...", 'b': "I would see this as an opportunity...", 'c': "I wouldn't do anything", 'd': "I would liquidate a part...", 'e': "I would liquidate the entire..."},
    'q13': {'a': "Above 1 million euros", 'b': "300,001 to 1 million euros", 'c': "80,001 to 300,000 euros", 'd': "30,001 to 80,000 euros", 'e': "Up to 30,000 euros"},
    'q6': {'a': "Low. It is not in my interests...", 'b': "Average. I occasionally update...", 'c': "Important. I regularly follow...", 'd': "High. I am constantly informed..."},
    'q7': {'a': "No or minimal experience...", 'b': "Moderate experience...", 'c': "Significant experience...", 'd': "Extensive experience..."},
    'q8': {'a': "Rarely (1-2 times a year)", 'b': "Occasional (1 time every 2-3 months)", 'c': "Often (1 time every fortnight or month)", 'd': "Very often (at least 2 times a week)"}
}
