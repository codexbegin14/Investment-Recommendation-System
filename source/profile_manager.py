import pandas as pd
from datetime import datetime



def process_questionnaire_responses(responses):

    # Risk level determination based on key questions
    risk_questions = {
        'q16': 0.3,  # Risk appetite
        'q17': 0.3,  # Investment expectations
        'q18': 0.2,  # Focus on gains vs losses
        'q19': 0.2   # Reaction to 20% decline
    }
    
    risk_score = 0
    # Calculate weighted risk score
    for q, weight in risk_questions.items():
        if q in responses and responses[q] is not None:
            answer = responses[q]
            # Mapping: 'a' is highest risk (4), 'e' is lowest risk (0)
            score_map = {'a': 4, 'b': 3, 'c': 2, 'd': 1, 'e': 0}
            if answer in score_map:
                risk_score += weight * score_map[answer]
    
    # Map risk score to risk level
    if risk_score >= 3.5:
        risk_level = "Aggressive"
    elif risk_score >= 2.5:
        risk_level = "Balanced"
    elif risk_score >= 1.5:
        risk_level = "Income"
    else:
        risk_level = "Conservative"
    
    # Investment capacity determination
    if 'q13' in responses and responses['q13'] is not None:
        investment = responses['q13']
        # Mapping: 'a' is highest capacity, 'e' is lowest capacity
        if investment == 'a':
            investment_capacity = "CAP_GT300K"
        elif investment == 'b':
            investment_capacity = "CAP_80K_300K"
        elif investment == 'c':
            investment_capacity = "CAP_30K_80K"
        else:
            investment_capacity = "CAP_LT30K"
    else:
        investment_capacity = "CAP_LT30K"  # Default to lowest capacity
    
    return risk_level, investment_capacity

def update_customer_profile(customer_id, risk_level, investment_capacity, customer_df):
  
    current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    new_row = pd.DataFrame({
        'customerID': [customer_id],
        # Assuming new users from the questionnaire start as 'Mass' type
        'customerType': ['Mass'], 
        'riskLevel': [risk_level],
        'investmentCapacity': [investment_capacity],
        'lastQuestionnaireDate': [current_time_str.split(' ')[0]],
        'timestamp': [current_time_str]
    })
    
    # Append new row to customer_df
    updated_df = pd.concat([customer_df, new_row], ignore_index=True)
    return updated_df