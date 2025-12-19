import pandas as pd
from datetime import datetime

def process_questionnaire_responses(responses):
    """
    Calculates Risk Level and Investment Capacity based on questionnaire answers.
    """
    # Risk level determination based on key questions
    risk_questions = {
        'Q1': 0.3,  # Risk appetite
        'Q2': 0.3,  # Investment expectations
        'Q3': 0.2,  # Focus on gains vs losses
        'Q4': 0.2   # Reaction to 20% decline
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
    
    # Investment capacity determination (matches questions.py options for Q5)
    capacity_map = {
        'a': "CAP_GT1M",       # >1M
        'b': "CAP_300K_1M",    # 300k-1M
        'c': "CAP_80K_300K",   # 80k-300k
        'd': "CAP_30K_80K",    # 30k-80k
        'e': "CAP_LT30K"       # <=30k
    }
    investment = responses.get('Q5')
    investment_capacity = capacity_map.get(investment, "CAP_LT30K")
    
    return risk_level, investment_capacity

def update_customer_profile(customer_id, risk_level, investment_capacity, customer_df):
    """
    Updates the customer dataframe.
    IMPORTANT: Maintains the Customer ID as the DataFrame Index so app.py can find it.
    """
    current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 1. Create a dictionary with the specific column names app.py expects
    # Note: app.py uses .get('Risk_Level') and .get('Investment_Capacity')
    data = {
        'customerID': [customer_id],
        'customerType': ['Mass'], 
        'Risk_Level': [risk_level],              # FIXED: Matches app.py expectation
        'Investment_Capacity': [investment_capacity], # FIXED: Matches app.py expectation
        'lastQuestionnaireDate': [current_time_str.split(' ')[0]],
        'timestamp': [current_time_str]
    }
    
    # 2. Create the new row and SET THE INDEX explicitly
    new_row = pd.DataFrame(data, index=[customer_id])
    
    # 3. Check if user already exists in the DataFrame (converting to string to be safe)
    # This prevents duplicate rows if they update their profile twice
    str_index = customer_df.index.astype(str)
    
    if str(customer_id) in str_index:
        # Update specific columns for the existing user
        # We find the location of this index and update it
        customer_df.loc[customer_df.index.astype(str) == str(customer_id), 'Risk_Level'] = risk_level
        customer_df.loc[customer_df.index.astype(str) == str(customer_id), 'Investment_Capacity'] = investment_capacity
        return customer_df
    else:
        # 4. Append new row
        # IMPORTANT: Removed ignore_index=True so we keep the customer_id as the index
        updated_df = pd.concat([customer_df, new_row])
        return updated_df