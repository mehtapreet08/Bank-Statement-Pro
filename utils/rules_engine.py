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