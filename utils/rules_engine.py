# ✅ rules_engine.py — Auto-recategorization of past transactions
import json
import re

class RulesEngine:
    def __init__(self):
        with open('data/custom_categories.json', 'r') as f:
            self.rules = json.load(f)

    def apply_rules_to_df(self, df):
        df['Category'] = df['Narration'].apply(self.get_category)
        return df

    def get_category(self, narration):
        for rule in self.rules:
            pattern = rule['pattern']
            category = rule['category']
            if re.search(pattern, narration, re.IGNORECASE):
                return category
        return 'Others'

    def update_rules(self, new_rule):
        self.rules.append(new_rule)
        with open('data/custom_categories.json', 'w') as f:
            json.dump(self.rules, f, indent=2)
        return True
import json
import re
import pandas as pd

class RulesEngine:
    def __init__(self):
        self.rules_file = 'data/custom_categories.json'
        self.load_rules()
    
    def load_rules(self):
        """Load categorization rules from file"""
        try:
            with open(self.rules_file, 'r') as f:
                self.rules = json.load(f)
        except:
            self.rules = []
    
    def apply_rules_to_df(self, df):
        """Apply all rules to a DataFrame"""
        df_copy = df.copy()
        
        for _, row in df_copy.iterrows():
            narration = str(row['narration']).upper()
            
            for rule in self.rules:
                if isinstance(rule, dict):
                    pattern = rule.get('pattern', '')
                    category = rule.get('category', 'Others')
                    
                    if re.search(pattern, narration, re.IGNORECASE):
                        df_copy.loc[df_copy['narration'] == row['narration'], 'category'] = category
                        break
        
        return df_copy
    
    def add_rule(self, pattern, category):
        """Add a new categorization rule"""
        new_rule = {"pattern": pattern, "category": category}
        self.rules.append(new_rule)
        
        with open(self.rules_file, 'w') as f:
            json.dump(self.rules, f, indent=2)
        
        return new_rulerule
