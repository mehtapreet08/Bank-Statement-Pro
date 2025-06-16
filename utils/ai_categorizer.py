
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
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # File paths
        self.categories_file = 'data/categories.json'
        self.custom_categories_file = 'data/custom_categories.json'
        self.cache_file = 'data/categorization_cache.json'
        self.category_types_file = 'data/category_types.json'
        self.default_categories_file = 'data/default_categories.json'
        
        # Initialize files if they don't exist
        self._initialize_data_files()
        
        # Load static data
        with open(self.categories_file, 'r') as f:
            self.categories = json.load(f)

        with open(self.custom_categories_file, 'r') as f:
            self.custom_categories = json.load(f)

        with open('data/category_types.json', 'r') as f:
            self.accounting_types = json.load(f)

        with open('data/default_categories.json', 'r') as f:
            self.default_categories = json.load(f)

        # Load dynamic patterns + learning
        self.categorization_cache = self._load_categorization_cache()
        
        # Load custom categories - handle both dict and list formats
        try:
            with open(self.custom_categories_file, 'r') as f:
                loaded_custom = json.load(f)
                # Keep the loaded format (could be list or dict)
                self.custom_categories = loaded_custom
        except:
            self.custom_categories = {}
            
        self.category_patterns = self._initialize_category_patterns()

        if not openai.api_key:
            print("⚠️ OpenAI API key not set. Falling back to rule-based categorization.")
    
    def _initialize_data_files(self):
        """Initialize all required JSON data files with default content"""
        
        # Categories file
        if not os.path.exists(self.categories_file):
            default_categories = {
                "Salary": ["salary", "wage", "payroll", "income", "pay"],
                "Food": ["restaurant", "food", "dining", "cafe", "kitchen", "meal", "swiggy", "zomato", "uber eats"],
                "Transportation": ["uber", "ola", "taxi", "bus", "metro", "fuel", "petrol", "diesel", "parking"],
                "Shopping": ["amazon", "flipkart", "myntra", "shopping", "purchase", "buy"],
                "Utilities": ["electricity", "water", "gas", "internet", "mobile", "phone", "broadband"],
                "Healthcare": ["hospital", "doctor", "medical", "pharmacy", "medicine", "health"],
                "Entertainment": ["movie", "netflix", "spotify", "game", "entertainment", "cinema"],
                "Investment": ["mutual fund", "sip", "stock", "investment", "dividend", "equity"],
                "EMI": ["emi", "loan", "mortgage", "credit", "installment"],
                "Others": ["misc", "other", "unknown"]
            }
            with open(self.categories_file, 'w') as f:
                json.dump(default_categories, f, indent=2)
        
        # Custom categories file
        if not os.path.exists(self.custom_categories_file):
            default_custom_rules = [
                {"pattern": "SALARY|WAGE|PAY.*ROLL", "category": "Salary"},
                {"pattern": "UPI.*SWIGGY|ZOMATO|UBER.*EATS", "category": "Food"},
                {"pattern": "DIVIDEND|DIV.*INCOME", "category": "Investment"},
                {"pattern": "EMI|LOAN.*EMI|HDFC.*LOAN", "category": "EMI"}
            ]
            with open(self.custom_categories_file, 'w') as f:
                json.dump(default_custom_rules, f, indent=2)
        
        # Category types file
        if not os.path.exists(self.category_types_file):
            category_types = {
                "Salary": "income",
                "Dividend": "income", 
                "Food": "expense",
                "Transportation": "expense",
                "Shopping": "expense",
                "Utilities": "expense",
                "Healthcare": "expense",
                "Entertainment": "expense",
                "Investment": "asset",
                "EMI": "liability",
                "Others": "expense"
            }
            with open(self.category_types_file, 'w') as f:
                json.dump(category_types, f, indent=2)
        
        # Default categories file
        if not os.path.exists(self.default_categories_file):
            default_categories_detailed = {
                "Salary": {"keywords": ["salary", "wage", "payroll", "income", "pay"], "type": "income"},
                "Food": {"keywords": ["restaurant", "food", "dining", "cafe", "kitchen", "meal", "swiggy", "zomato", "uber eats"], "type": "expense"},
                "Transportation": {"keywords": ["uber", "ola", "taxi", "bus", "metro", "fuel", "petrol", "diesel", "parking"], "type": "expense"},
                "Shopping": {"keywords": ["amazon", "flipkart", "myntra", "shopping", "purchase", "buy"], "type": "expense"},
                "Utilities": {"keywords": ["electricity", "water", "gas", "internet", "mobile", "phone", "broadband"], "type": "expense"},
                "Healthcare": {"keywords": ["hospital", "doctor", "medical", "pharmacy", "medicine", "health"], "type": "expense"},
                "Entertainment": {"keywords": ["movie", "netflix", "spotify", "game", "entertainment", "cinema"], "type": "expense"},
                "Investment": {"keywords": ["mutual fund", "sip", "stock", "investment", "dividend", "equity"], "type": "asset"},
                "EMI": {"keywords": ["emi", "loan", "mortgage", "credit", "installment"], "type": "liability"},
                "Others": {"keywords": ["misc", "other", "unknown"], "type": "expense"}
            }
            with open(self.default_categories_file, 'w') as f:
                json.dump(default_categories_detailed, f, indent=2)

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
        
        # Process default categories
        for category, category_data in self.default_categories.items():
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
        
        # Process custom categories (if they exist and are in dict format)
        if isinstance(self.custom_categories, dict):
            for category, category_data in self.custom_categories.items():
                if isinstance(category_data, dict):
                    keywords = category_data.get('keywords', [])
                else:
                    keywords = category_data
                
                pattern_strings = []
                for keyword in keywords:
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
        # Convert to dict format if currently list format
        if not isinstance(self.custom_categories, dict):
            self.custom_categories = {}
        
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
        try:
            from .data_manager import DataManager
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
        if isinstance(self.custom_categories, dict) and category_name in self.custom_categories:
            del self.custom_categories[category_name]
            self._save_custom_categories()
            
            # Update category patterns
            self.category_patterns = self._initialize_category_patterns()
        elif isinstance(self.custom_categories, list):
            # Remove all rules for this category
            self.custom_categories = [rule for rule in self.custom_categories 
                                    if rule.get('category') != category_name]
            self._save_custom_categories()
            
            # Update category patterns
            self.category_patterns = self._initialize_category_patterns()
    
    def get_all_categories(self) -> List[str]:
        """Get list of all available categories"""
        default_cats = list(self.default_categories.keys())
        
        # Handle custom categories - could be dict or list format
        if isinstance(self.custom_categories, dict):
            custom_cats = list(self.custom_categories.keys())
        elif isinstance(self.custom_categories, list):
            # If it's a list of rules, extract unique categories
            custom_cats = list(set([rule.get('category', 'Others') for rule in self.custom_categories if isinstance(rule, dict)]))
        else:
            custom_cats = []
        
        return default_cats + custom_cats
    
    def get_default_categories(self) -> List[str]:
        """Get list of default categories"""
        return list(self.default_categories.keys())
    
    def get_custom_categories(self) -> Dict[str, Dict]:
        """Get custom categories with their keywords and types"""
        if isinstance(self.custom_categories, dict):
            return self.custom_categories.copy()
        else:
            # Convert list format to dict format for display
            result = {}
            for rule in self.custom_categories:
                if isinstance(rule, dict) and 'category' in rule:
                    category = rule['category']
                    if category not in result:
                        result[category] = {
                            'keywords': [rule.get('pattern', '')],
                            'type': 'expense'
                        }
                    else:
                        result[category]['keywords'].append(rule.get('pattern', ''))
            return result
    
    def get_category_type(self, category_name: str) -> str:
        """Get the type of a specific category"""
        if category_name in self.default_categories:
            return self.default_categories[category_name].get('type', 'expense')
        elif isinstance(self.custom_categories, dict) and category_name in self.custom_categories:
            return self.custom_categories[category_name].get('type', 'expense')
        elif isinstance(self.custom_categories, list):
            # Find first rule with this category
            for rule in self.custom_categories:
                if isinstance(rule, dict) and rule.get('category') == category_name:
                    return rule.get('type', 'expense')
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
