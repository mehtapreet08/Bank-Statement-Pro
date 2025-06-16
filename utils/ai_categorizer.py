import json
import os
import re
import pickle
from typing import Dict, List, Optional, Tuple
import pandas as pd
from fuzzywuzzy import fuzz
from datetime import datetime
import numpy as np
import openai

        # Configure OpenRouter
openai.api_key = "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"
openai.api_base = "https://openrouter.ai/api/v1"

class AICategorizer:
    def __init__(self):
        # File paths
        self.categories_file = 'data/categories.json'
        self.custom_categories_file = 'data/custom_categories.json'
        self.cache_file = 'data/categorization_cache.json'
        
        # Load static data
        with open(self.categories_file, 'r') as f:
            self.categories = json.load(f)

        with open(self.custom_categories_file, 'r') as f:
            self.custom_rules = json.load(f)

        with open('data/category_types.json', 'r') as f:
            self.accounting_types = json.load(f)

        with open('data/default_categories.json', 'r') as f:
            self.default_categories = json.load(f)

        # Load dynamic patterns + learning
        self.categorization_cache = self._load_categorization_cache()
        self.custom_categories = self._load_custom_categories()
        self.default_categories = self._load_default_categories()
        self.category_patterns = self._initialize_category_patterns()
        self.default_categories = self._load_default_categories()

        if not openai.api_key:
            print("⚠️ OpenAI API key not set. Falling back to rule-based categorization.")

            with open('data/categories.json', 'r') as f:
                self.categories = json.load(f)

            with open('data/custom_categories.json', 'r') as f:
                self.custom_rules = json.load(f)

            with open('data/category_types.json', 'r') as f:
                self.accounting_types = json.load(f)
                self.categorization_cache = self._load_categorization_cache()


    # def categorize_transactions(self, df):
    #     df['Category'] = df.apply(lambda row: self.apply_rules(row['Narration']), axis=1)
    #     df['CategoryType'] = df['Category'].map(self.accounting_types).fillna('Unknown')
    #     return df

    def apply_rules(self, narration):
        for rule in self.custom_rules:
            pattern = rule['pattern']
            category = rule['category']
            if re.search(pattern, narration, re.IGNORECASE):
                return category

        return self.ai_guess_category(narration)

    def ai_guess_category(self, narration):
        prompt = f"""
        Classify this bank transaction narration into one of the following categories:
        {', '.join(self.categories)}

        Narration: "{narration}"
        Return only the category name.
        """

        try:
            response = openai.ChatCompletion.create(
                model="google/gemma-3n-e4b-it:free",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=10
            )
            result = response['choices'][0]['message']['content'].strip()
            return result if result in self.categories else 'Others'
        except Exception as e:
            print(f"AI Error: {e}")
            return 'Others'

        
    def _load_default_categories(self) -> Dict[str, Dict]:
        """Load default categories from JSON file, fallback to minimal default if missing"""
        default_path = "data/default_categories.json"

        if os.path.exists(default_path):
            try:
                with open(default_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                raise Exception(f"❌ Failed to read default categories: {str(e)}")

        # Fallback: return minimal category if file is missing or corrupted
        return {
            "Others": {
                "keywords": ["misc", "other", "unknown"],
                "type": "expense"
            }
        }

    
    def _load_custom_categories(self) -> Dict[str, Dict]:
        """Load user-defined custom categories"""
        if os.path.exists(self.custom_categories_file):
            try:
                with open(self.custom_categories_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def _load_categorization_cache(self) -> Dict[str, Dict]:
        """Load the permanent categorization learning cache"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {
            "patterns": {},  # narration -> category mappings
            "corrections": {},  # user corrections for learning
            "fuzzy_matches": {},  # fuzzy match results
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_categorization_cache(self):
        """Save the categorization cache permanently"""
        self.categorization_cache["last_updated"] = datetime.now().isoformat()
        with open(self.cache_file, 'w') as f:
            json.dump(self.categorization_cache, f, indent=2)
    
    def _save_custom_categories(self):
        """Save custom categories to file"""
        with open(self.custom_categories_file, 'w') as f:
            json.dump(self.custom_categories, f, indent=2)
    
    def _initialize_category_patterns(self) -> Dict:
        """Initialize regex patterns for each category"""
        patterns = {}
        
        # Combine default and custom categories
        all_categories = {**self.default_categories, **self.custom_categories}
        
        for category, category_data in all_categories.items():
            # Handle both old and new format
            if isinstance(category_data, dict):
                keywords = category_data.get('keywords', [])
            else:
                keywords = category_data  # For backwards compatibility
            
            # Create regex pattern for each category
            pattern_strings = []
            for keyword in keywords:
                # Escape special regex characters and add word boundaries
                escaped_keyword = re.escape(keyword.lower())
                pattern_strings.append(f"\\b{escaped_keyword}\\b")
            
            if pattern_strings:
                patterns[category] = re.compile('|'.join(pattern_strings), re.IGNORECASE)
        
        return patterns
    
    def categorize_transactions(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Categorize transactions using AI with permanent learning
        
        Args:
            transactions_df: DataFrame with transaction data
            
        Returns:
            DataFrame with added 'category' and 'ai_categorized' columns
        """
        df = transactions_df.copy()
        
        # Add columns for category and AI flag
        categorization_results = df['narration'].apply(self._categorize_single_transaction_with_flag)
        df['category'] = [result[0] for result in categorization_results]
        df['ai_categorized'] = [result[1] for result in categorization_results]
        
        # Save any new patterns learned during categorization
        self._save_categorization_cache()
        
        return df
    
    def _categorize_single_transaction_with_flag(self, narration: str) -> tuple:
        """Categorize a single transaction and return if it was AI categorized"""
        narration_lower = narration.lower().strip()
        
        # 1. Check exact cache matches first (highest priority) - not AI
        if narration_lower in self.categorization_cache["patterns"]:
            return self.categorization_cache["patterns"][narration_lower], False
        
        # 2. Check user corrections cache - not AI
        if narration_lower in self.categorization_cache["corrections"]:
            return self.categorization_cache["corrections"][narration_lower], False
        
        # 3. Apply rule-based categorization - this is AI
        category = self._apply_rule_based_categorization(narration_lower)
        
        # 4. If no rule match, try fuzzy matching against cached patterns - this is AI
        if category == "Others":
            fuzzy_category = self._apply_fuzzy_matching(narration_lower)
            if fuzzy_category:
                category = fuzzy_category
        
        # 5. Cache the result for future use
        self.categorization_cache["patterns"][narration_lower] = category
        
        # Return category and AI flag (True if it was AI categorized)
        return category, True
    
    def _categorize_single_transaction(self, narration: str) -> str:
        """Categorize a single transaction narration"""
        narration_lower = narration.lower().strip()
        
        # 1. Check exact cache matches first (highest priority)
        if narration_lower in self.categorization_cache["patterns"]:
            return self.categorization_cache["patterns"][narration_lower]
        
        # 2. Check user corrections cache
        if narration_lower in self.categorization_cache["corrections"]:
            return self.categorization_cache["corrections"][narration_lower]
        
        # 3. Apply rule-based categorization
        category = self._apply_rule_based_categorization(narration_lower)
        
        # 4. If no rule match, try fuzzy matching against cached patterns
        if category == "Others":
            fuzzy_category = self._apply_fuzzy_matching(narration_lower)
            if fuzzy_category:
                category = fuzzy_category
        
        # 5. Cache the result for future use
        self.categorization_cache["patterns"][narration_lower] = category
        
        return category
    
    def _apply_rule_based_categorization(self, narration: str) -> str:
        """Apply rule-based categorization using keyword patterns"""
        # Check each category pattern
        for category, pattern in self.category_patterns.items():
            if pattern.search(narration):
                return category
        
        # Special rules for amount-based categorization
        return self._apply_special_rules(narration)
    
    def _apply_special_rules(self, narration: str) -> str:
        """Apply special categorization rules"""
        # ATM withdrawals
        if any(term in narration for term in ["atm", "cash withdrawal", "pos"]):
            return "Cash Withdrawal"
        
        # Bank transfers
        if any(term in narration for term in ["neft", "rtgs", "imps", "upi", "transfer"]):
            return "Transfer"
        
        # Interest and charges
        if any(term in narration for term in ["interest", "charge", "fee", "penalty"]):
            return "Bank Charges"
        
        return "Others"
    
    def _apply_fuzzy_matching(self, narration: str, threshold: int = 80) -> Optional[str]:
        """Apply fuzzy matching against cached patterns"""
        best_match_score = 0
        best_match_category = None
        
        # Check against all cached patterns
        for cached_narration, category in self.categorization_cache["patterns"].items():
            score = fuzz.ratio(narration, cached_narration)
            if score > threshold and score > best_match_score:
                best_match_score = score
                best_match_category = category
        
        # Check against user corrections with higher weight
        for cached_narration, category in self.categorization_cache["corrections"].items():
            score = fuzz.ratio(narration, cached_narration)
            if score > (threshold - 10) and score > best_match_score:  # Lower threshold for corrections
                best_match_score = score
                best_match_category = category
        
        if best_match_category:
            # Cache the fuzzy match result
            self.categorization_cache["fuzzy_matches"][narration] = {
                "category": best_match_category,
                "score": best_match_score,
                "matched_against": cached_narration
            }
        
        return best_match_category
    
    def learn_from_correction(self, narration: str, correct_category: str):
        """
        Learn from user corrections to improve future categorization
        
        Args:
            narration: Transaction narration
            correct_category: User-corrected category
        """
        narration_lower = narration.lower().strip()
        
        # Store the correction in the learning cache
        self.categorization_cache["corrections"][narration_lower] = correct_category
        
        # Also update the main patterns cache
        self.categorization_cache["patterns"][narration_lower] = correct_category
        
        # Extract keywords from the narration for pattern enhancement
        self._enhance_category_patterns(narration_lower, correct_category)
        
        # Save the updated cache
        self._save_categorization_cache()
    
    def _enhance_category_patterns(self, narration: str, category: str):
        """Enhance category patterns based on user corrections"""
        # Extract potential keywords from the narration
        words = re.findall(r'\b\w+\b', narration.lower())
        
        # Filter out common words and numbers
        stop_words = {'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if len(word) > 2 and word not in stop_words and not word.isdigit()]
        
        # Add significant keywords to the category if not already present
        if category in self.default_categories:
            existing_keywords = [kw.lower() for kw in self.default_categories[category]]
            new_keywords = [kw for kw in keywords if kw not in existing_keywords]
            
            if new_keywords:
                # Don't automatically add to default categories, but store in cache
                cache_key = f"learned_keywords_{category}"
                if cache_key not in self.categorization_cache:
                    self.categorization_cache[cache_key] = []
                self.categorization_cache[cache_key].extend(new_keywords)
    
    def add_custom_category(self, category_name: str, keywords: List[str], category_type: str = "expense"):
        """Add a new custom category with keywords and type"""
        self.custom_categories[category_name] = {
            "keywords": keywords,
            "type": category_type
        }
        self._save_custom_categories()
        
        # Update category patterns
        self.category_patterns = self._initialize_category_patterns()
        
        # Re-categorize existing transactions if needed
        self._recategorize_existing_transactions()
    
    def _recategorize_existing_transactions(self):
        """Re-categorize existing transactions when new rules are added"""
        from .data_manager import DataManager
        
        try:
            data_mgr = DataManager()
            existing_df = data_mgr.load_transactions()
            
            if not existing_df.empty:
                # Re-categorize all transactions
                updated_df = self.categorize_transactions(existing_df)
                data_mgr.save_transactions(updated_df)
        except Exception as e:
            print(f"Error re-categorizing existing transactions: {str(e)}")
    
    def delete_custom_category(self, category_name: str):
        """Delete a custom category"""
        if category_name in self.custom_categories:
            del self.custom_categories[category_name]
            self._save_custom_categories()
            
            # Update category patterns
            self.category_patterns = self._initialize_category_patterns()
    
    def get_all_categories(self) -> List[str]:
        """Get list of all available categories"""
        return list(self.default_categories.keys()) + list(self.custom_categories.keys())
    
    def get_default_categories(self) -> List[str]:
        """Get list of default categories"""
        return list(self.default_categories.keys())
    
    def get_custom_categories(self) -> Dict[str, Dict]:
        """Get custom categories with their keywords and types"""
        return self.custom_categories.copy()
    
    def get_category_type(self, category_name: str) -> str:
        """Get the type of a specific category"""
        if category_name in self.default_categories:
            return self.default_categories[category_name].get('type', 'expense')
        elif category_name in self.custom_categories:
            return self.custom_categories[category_name].get('type', 'expense')
        return 'expense'
    
    def get_cache_statistics(self) -> Dict:
        """Get statistics about the categorization cache"""
        try:
            cache_size = os.path.getsize(self.cache_file) if os.path.exists(self.cache_file) else 0
            cache_size_kb = round(cache_size / 1024, 2)
            
            return {
                "pattern_count": len(self.categorization_cache.get("patterns", {})),
                "correction_count": len(self.categorization_cache.get("corrections", {})),
                "fuzzy_match_count": len(self.categorization_cache.get("fuzzy_matches", {})),
                "cache_size_kb": cache_size_kb,
                "last_updated": self.categorization_cache.get("last_updated", "Never")
            }
        except Exception:
            return {
                "pattern_count": 0,
                "correction_count": 0,
                "fuzzy_match_count": 0,
                "cache_size_kb": 0,
                "last_updated": "Never"
            }
    
    def clear_cache(self):
        """Clear the categorization cache"""
        self.categorization_cache = {
            "patterns": {},
            "corrections": {},
            "fuzzy_matches": {},
            "last_updated": datetime.now().isoformat()
        }
        self._save_categorization_cache()
    
    def export_cache(self) -> str:
        """Export categorization cache as JSON string"""
        return json.dumps(self.categorization_cache, indent=2)
    
    def import_cache(self, cache_json: str):
        """Import categorization cache from JSON string"""
        try:
            imported_cache = json.loads(cache_json)
            
            # Merge with existing cache
            for key in ["patterns", "corrections", "fuzzy_matches"]:
                if key in imported_cache:
                    self.categorization_cache[key].update(imported_cache[key])
            
            self._save_categorization_cache()
            return True
        except Exception:
            return False
