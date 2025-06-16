# ✅ category_manager.py — Add + Apply Rule Immediately
import json
import pandas as pd
from utils.rules_engine import RulesEngine

class CategoryManager:
    def __init__(self):
        self.rules_engine = RulesEngine()
        with open('data/custom_categories.json', 'r') as f:
            self.rules = json.load(f)

    def add_rule(self, pattern, category):
        new_rule = {"pattern": pattern, "category": category}
        self.rules.append(new_rule)
        with open('data/custom_categories.json', 'w') as f:
            json.dump(self.rules, f, indent=2)
        return new_rule

    def add_and_apply_rule(self, pattern, category, df):
        self.add_rule(pattern, category)
        df = self.rules_engine.apply_rules_to_df(df)
        return df