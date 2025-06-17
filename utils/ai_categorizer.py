import json
import os
import re
import pickle
from typing import Dict, List, Optional, Tuple
import pandas as pd
from fuzzywuzzy import fuzz
from datetime import datetime
import numpy as np
from together import Together

# Configure Together AI
TOGETHER_API_KEY = "d506e2a8ffe9aff1839a2587587a6c2b78c8852867bfbcb4daa3dbc8676e4650"


class AICategorizer:

    def __init__(self, user_data_dir=None):
        # Use user-specific data directory or default to 'data'
        self.data_dir = user_data_dir if user_data_dir else 'data'

        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)

        # File paths
        self.categories_file = os.path.join(self.data_dir, 'categories.json')
        self.custom_categories_file = os.path.join(self.data_dir,
                                                   'custom_categories.json')
        self.cache_file = os.path.join(self.data_dir,
                                       'categorization_cache.json')
        self.category_types_file = os.path.join(self.data_dir,
                                                'category_types.json')
        self.default_categories_file = os.path.join(self.data_dir,
                                                    'default_categories.json')

        # Initialize files if they don't exist
        self._initialize_data_files()

        # Load static data
        with open(self.categories_file, 'r') as f:
            self.categories = json.load(f)

        with open(self.custom_categories_file, 'r') as f:
            self.custom_categories = json.load(f)

        with open(self.category_types_file, 'r') as f:
            self.accounting_types = json.load(f)

        with open(self.default_categories_file, 'r') as f:
            self.default_categories = json.load(f)

        # Load dynamic patterns + learning
        self.categorization_cache = self._load_categorization_cache()

        # Load custom categories - handle both dict and list formats
        try:
            with open(self.custom_categories_file, 'r') as f:
                loaded_custom = json.load(f)
                # Keep the loaded format (could be list or dict)
                self.custom_categories = loaded_custom
                print(
                    f"Debug: Loaded custom categories: {self.custom_categories}"
                )
        except Exception as e:
            print(f"Debug: Error loading custom categories: {e}")
            self.custom_categories = {}

        self.category_patterns = self._initialize_category_patterns()

        if not TOGETHER_API_KEY:
            print(
                "⚠️ Together AI API key not set. Falling back to rule-based categorization."
            )

    def _initialize_data_files(self):
        """Initialize all required JSON data files with default content"""

        # Categories file
        if not os.path.exists(self.categories_file):
            default_categories = {
                "Salary": ["salary", "wage", "payroll", "income", "pay"],
                "Food": [
                    "restaurant", "food", "dining", "cafe", "kitchen", "meal",
                    "swiggy", "zomato", "uber eats"
                ],
                "Transportation": [
                    "uber", "ola", "taxi", "bus", "metro", "fuel", "petrol",
                    "diesel", "parking"
                ],
                "Shopping": [
                    "amazon", "flipkart", "myntra", "shopping", "purchase",
                    "buy"
                ],
                "Utilities": [
                    "electricity", "water", "gas", "internet", "mobile",
                    "phone", "broadband"
                ],
                "Healthcare": [
                    "hospital", "doctor", "medical", "pharmacy", "medicine",
                    "health"
                ],
                "Entertainment": [
                    "movie", "netflix", "spotify", "game", "entertainment",
                    "cinema"
                ],
                "Investment": [
                    "mutual fund", "sip", "stock", "investment", "dividend",
                    "equity"
                ],
                "EMI": ["emi", "loan", "mortgage", "credit", "installment"],
                "Suspense": ["suspense", "uncertain", "low confidence"],
                "Others": ["misc", "other", "unknown"]
            }
            with open(self.categories_file, 'w') as f:
                json.dump(default_categories, f, indent=2)

        # Custom categories file
        if not os.path.exists(self.custom_categories_file):
            default_custom_rules = [{
                "pattern": "SALARY|WAGE|PAY.*ROLL",
                "category": "Salary"
            }, {
                "pattern": "UPI.*SWIGGY|ZOMATO|UBER.*EATS",
                "category": "Food"
            }, {
                "pattern": "DIVIDEND|DIV.*INCOME",
                "category": "Investment"
            }, {
                "pattern": "EMI|LOAN.*EMI|HDFC.*LOAN",
                "category": "EMI"
            }]
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
                "Suspense": "expense",
                "Others": "expense"
            }
            with open(self.category_types_file, 'w') as f:
                json.dump(category_types, f, indent=2)

        # Default categories file
        if not os.path.exists(self.default_categories_file):
            default_categories_detailed = {
                "Salary": {
                    "keywords": ["salary", "wage", "payroll", "income", "pay"],
                    "type": "income"
                },
                "Food": {
                    "keywords": [
                        "restaurant", "food", "dining", "cafe", "kitchen",
                        "meal", "swiggy", "zomato", "uber eats"
                    ],
                    "type":
                    "expense"
                },
                "Transportation": {
                    "keywords": [
                        "uber", "ola", "taxi", "bus", "metro", "fuel",
                        "petrol", "diesel", "parking"
                    ],
                    "type":
                    "expense"
                },
                "Shopping": {
                    "keywords": [
                        "amazon", "flipkart", "myntra", "shopping", "purchase",
                        "buy"
                    ],
                    "type":
                    "expense"
                },
                "Utilities": {
                    "keywords": [
                        "electricity", "water", "gas", "internet", "mobile",
                        "phone", "broadband"
                    ],
                    "type":
                    "expense"
                },
                "Healthcare": {
                    "keywords": [
                        "hospital", "doctor", "medical", "pharmacy",
                        "medicine", "health"
                    ],
                    "type":
                    "expense"
                },
                "Entertainment": {
                    "keywords": [
                        "movie", "netflix", "spotify", "game", "entertainment",
                        "cinema"
                    ],
                    "type":
                    "expense"
                },
                "Investment": {
                    "keywords": [
                        "mutual fund", "sip", "stock", "investment",
                        "dividend", "equity"
                    ],
                    "type":
                    "asset"
                },
                "EMI": {
                    "keywords":
                    ["emi", "loan", "mortgage", "credit", "installment"],
                    "type":
                    "liability"
                },
                "Suspense": {
                    "keywords": ["suspense", "uncertain", "low confidence"],
                    "type": "expense"
                },
                "Others": {
                    "keywords": ["misc", "other", "unknown"],
                    "type": "expense"
                }
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
            client = Together(api_key=TOGETHER_API_KEY)

            response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.2,
                max_tokens=10)

            result = response.choices[0].message.content.strip()
            return result if result in self.categories else 'Others'
        except Exception as e:
            print(f"AI Error: {e}")
            return 'Others'

    def _load_default_categories(self) -> Dict[str, Dict]:
        """Load default categories from JSON file, fallback to minimal default if missing"""
        default_path = self.default_categories_file

        if os.path.exists(default_path):
            try:
                with open(default_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                raise Exception(
                    f"❌ Failed to read default categories: {str(e)}")

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
                if keyword:  # Skip empty keywords
                    # Escape special regex characters but allow partial matches
                    escaped_keyword = re.escape(str(keyword).lower())
                    # Use more flexible matching - not just word boundaries
                    pattern_strings.append(f"{escaped_keyword}")

            if pattern_strings:
                patterns[category] = re.compile('|'.join(pattern_strings),
                                                re.IGNORECASE)

        # Process custom categories (if they exist and are in dict format)
        if isinstance(self.custom_categories, dict):
            for category, category_data in self.custom_categories.items():
                if isinstance(category_data, dict):
                    keywords = category_data.get('keywords', [])
                    # Ensure keywords is a list
                    if not isinstance(keywords, list):
                        keywords = [keywords] if keywords else []
                else:
                    # Handle old format
                    keywords = category_data if isinstance(
                        category_data, list) else [category_data]

                pattern_strings = []
                for keyword in keywords:
                    if keyword:  # Skip empty keywords
                        # Escape special regex characters but allow partial matches
                        escaped_keyword = re.escape(str(keyword).lower())
                        # Use more flexible matching - not just word boundaries
                        pattern_strings.append(f"{escaped_keyword}")

                if pattern_strings:
                    patterns[category] = re.compile('|'.join(pattern_strings),
                                                    re.IGNORECASE)

        return patterns

    def categorize_transactions(self,
                                transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Categorize transactions using AI with permanent learning

        Args:
            transactions_df: DataFrame with transaction data

        Returns:
            DataFrame with added 'category', 'ai_categorized', and 'similarity_score' columns
        """
        df = transactions_df.copy()

        # Add columns for category, AI flag, and similarity score
        categorization_results = df['narration'].apply(
            self._categorize_single_transaction_with_similarity)
        df['category'] = [result[0] for result in categorization_results]
        df['ai_categorized'] = [result[1] for result in categorization_results]
        df['similarity_score'] = [
            result[2] for result in categorization_results
        ]

        # Save any new patterns learned during categorization
        self._save_categorization_cache()

        return df

    def _categorize_single_transaction_with_similarity(
            self, narration: str) -> tuple:
        """Categorize a single transaction and return category, AI flag, and similarity score"""
        narration_lower = narration.lower().strip()
        similarity_score = 0

        # 1. Check exact cache matches first (highest priority) - not AI
        if narration_lower in self.categorization_cache["patterns"]:
            return self.categorization_cache["patterns"][
                narration_lower], False, 100

        # 2. Check user corrections cache - not AI
        if narration_lower in self.categorization_cache["corrections"]:
            return self.categorization_cache["corrections"][
                narration_lower], False, 100

        # 3. Apply similarity-based categorization first
        category, similarity_score = self._apply_similarity_based_categorization(
            narration_lower)

        # 4. If no similarity match, apply rule-based categorization
        if category == "Others":
            category = self._apply_rule_based_categorization(narration_lower)
            similarity_score = 0

        # 5. If no rule match, try fuzzy matching against cached patterns
        if category == "Others":
            fuzzy_category = self._apply_fuzzy_matching(narration_lower)
            if fuzzy_category:
                category = fuzzy_category
                similarity_score = 80  # Assume 80% for fuzzy matches

        # 6. Cache the result for future use
        self.categorization_cache["patterns"][narration_lower] = category

        # Check if confidence is too low (less than 30%) and put in suspense
        if similarity_score < 30 and category != "Others":
            category = "Suspense"

        # Return category, AI flag, and similarity score
        return category, True, similarity_score

    def _categorize_single_transaction_with_flag(self,
                                                 narration: str) -> tuple:
        """Categorize a single transaction and return if it was AI categorized"""
        narration_lower = narration.lower().strip()

        # 1. Check exact cache matches first (highest priority) - not AI
        if narration_lower in self.categorization_cache["patterns"]:
            return self.categorization_cache["patterns"][
                narration_lower], False

        # 2. Check user corrections cache - not AI
        if narration_lower in self.categorization_cache["corrections"]:
            return self.categorization_cache["corrections"][
                narration_lower], False

        # 3. Apply similarity-based categorization first
        category, similarity_score = self._apply_similarity_based_categorization(
            narration_lower)

        # 4. If no similarity match, apply rule-based categorization
        if category == "Others":
            category = self._apply_rule_based_categorization(narration_lower)

        # 5. If no rule match, try fuzzy matching against cached patterns
        if category == "Others":
            fuzzy_category = self._apply_fuzzy_matching(narration_lower)
            if fuzzy_category:
                category = fuzzy_category

        # 6. Cache the result for future use
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

    def _apply_similarity_based_categorization(self, narration: str) -> tuple:
        """Apply similarity-based categorization using keyword matching"""
        best_category = "Others"
        best_similarity = 0

        # Check against all categories (default + custom)
        all_categories = {}

        # Add default categories
        for category, category_data in self.default_categories.items():
            if isinstance(category_data, dict):
                keywords = category_data.get('keywords', [])
            else:
                keywords = category_data
            all_categories[category] = keywords

        # Add custom categories - ensure proper format handling
        if isinstance(self.custom_categories, dict):
            for category, category_data in self.custom_categories.items():
                if isinstance(category_data, dict):
                    keywords = category_data.get('keywords', [])
                    # Ensure keywords is a list
                    if not isinstance(keywords, list):
                        keywords = [keywords] if keywords else []
                else:
                    # Handle old format
                    keywords = category_data if isinstance(
                        category_data, list) else [category_data]

                # Only add if we have valid keywords
                if keywords and any(kw for kw in keywords if kw):
                    all_categories[category] = keywords
        elif isinstance(self.custom_categories, list):
            # Handle old list format with patterns
            for rule in self.custom_categories:
                if isinstance(rule, dict) and 'category' in rule:
                    category = rule['category']
                    pattern = rule.get('pattern', '')
                    if pattern:
                        if category not in all_categories:
                            all_categories[category] = [pattern]
                        else:
                            all_categories[category].append(pattern)

        # Calculate similarity for each category
        for category, keywords in all_categories.items():
            if not keywords:  # Skip empty keyword lists
                continue

            for keyword in keywords:
                if not keyword:  # Skip empty keywords
                    continue

                similarity = self._calculate_similarity(
                    narration,
                    str(keyword).lower())
                # Debug: Print matching details for troubleshooting
                if similarity > 0:
                    print(
                        f"Debug: '{narration}' vs '{keyword}' in category '{category}' = {similarity}%"
                    )

                if similarity > best_similarity and similarity >= 20:  # Minimum 20% similarity
                    best_similarity = similarity
                    best_category = category

        return best_category, best_similarity

    def _calculate_similarity(self, narration: str, keyword: str) -> int:
        """Calculate similarity percentage between narration and keyword"""
        if not keyword or not narration:
            return 0

        narration_lower = narration.lower().strip()
        keyword_lower = keyword.lower().strip()

        # Remove special characters and normalize spaces
        narration_clean = re.sub(r'[^\w\s]', ' ', narration_lower)
        keyword_clean = re.sub(r'[^\w\s]', ' ', keyword_lower)

        narration_words = set(narration_clean.split())
        keyword_words = set(keyword_clean.split())

        if not keyword_words:
            return 0

        # Check for exact keyword match in narration (case-insensitive)
        if keyword_lower in narration_lower:
            return 100

        # Check for exact match after cleaning
        if keyword_clean in narration_clean:
            return 100

        # Check for individual keyword word matches in narration
        for keyword_word in keyword_words:
            if keyword_word in narration_lower:
                return 100

        # Check for word overlap
        common_words = narration_words & keyword_words
        if common_words:
            similarity = round((len(common_words) / len(keyword_words)) * 100)
            return max(similarity, 85)  # Give higher weight to word matches

        # Check for partial word matches (case-insensitive)
        for narration_word in narration_words:
            for keyword_word in keyword_words:
                if len(keyword_word) >= 3:  # Only for meaningful words
                    if keyword_word in narration_word or narration_word in keyword_word:
                        return 80  # Partial match

        # Check for substring matches in either direction
        for keyword_word in keyword_words:
            if len(keyword_word) >= 3:
                if keyword_word in narration_clean or any(
                        keyword_word in word for word in narration_words):
                    return 75

        return 0

    def _apply_special_rules(self, narration: str) -> str:
        """Apply special categorization rules"""
        # ATM withdrawals
        if any(term in narration
               for term in ["atm", "cash withdrawal", "pos"]):
            return "Cash Withdrawal"

        # Bank transfers
        if any(term in narration
               for term in ["neft", "rtgs", "imps", "upi", "transfer"]):
            return "Transfer"

        # Interest and charges
        if any(term in narration
               for term in ["interest", "charge", "fee", "penalty"]):
            return "Bank Charges"

        return "Others"

    def _apply_fuzzy_matching(self,
                              narration: str,
                              threshold: int = 80) -> Optional[str]:
        """Apply fuzzy matching against cached patterns"""
        best_match_score = 0
        best_match_category = None

        # Check against all cached patterns
        for cached_narration, category in self.categorization_cache[
                "patterns"].items():
            score = fuzz.ratio(narration, cached_narration)
            if score > threshold and score > best_match_score:
                best_match_score = score
                best_match_category = category

        # Check against user corrections with higher weight
        for cached_narration, category in self.categorization_cache[
                "corrections"].items():
            score = fuzz.ratio(narration, cached_narration)
            if score > (
                    threshold - 10
            ) and score > best_match_score:  # Lower threshold for corrections
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
        self.categorization_cache["corrections"][
            narration_lower] = correct_category

        # Also update the main patterns cache
        self.categorization_cache["patterns"][
            narration_lower] = correct_category

        # Extract keywords from the narration for pattern enhancement
        self._enhance_category_patterns(narration_lower, correct_category)

        # Save the updated cache
        self._save_categorization_cache()

    def _enhance_category_patterns(self, narration: str, category: str):
        """Enhance category patterns based on user corrections"""
        # Extract potential keywords from the narration
        words = re.findall(r'\b\w+\b', narration.lower())

        # Filter out common words and numbers
        stop_words = {
            'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by'
        }
        keywords = [
            word for word in words
            if len(word) > 2 and word not in stop_words and not word.isdigit()
        ]

        # Add significant keywords to the category if not already present
        if category in self.default_categories:
            existing_keywords = [
                kw.lower() for kw in self.default_categories[category]
            ]
            new_keywords = [
                kw for kw in keywords if kw not in existing_keywords
            ]

            if new_keywords:
                # Don't automatically add to default categories, but store in cache
                cache_key = f"learned_keywords_{category}"
                if cache_key not in self.categorization_cache:
                    self.categorization_cache[cache_key] = []
                self.categorization_cache[cache_key].extend(new_keywords)

    def add_custom_category(self,
                            category_name: str,
                            keywords: List[str],
                            category_type: str = "expense"):
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

    def update_custom_category(self,
                               category_name: str,
                               keywords: List[str],
                               category_type: str = "expense"):
        """Update an existing custom category"""
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

    def update_default_category(self,
                                category_name: str,
                                keywords: List[str],
                                category_type: str = "expense"):
        """Update an existing default category"""
        if category_name in self.default_categories:
            self.default_categories[category_name] = {
                "keywords": keywords,
                "type": category_type
            }
            self._save_default_categories()

            # Update category patterns
            self.category_patterns = self._initialize_category_patterns()

    def _save_default_categories(self):
        """Save default categories to file"""
        try:
            with open(self.default_categories_file, 'w') as f:
                json.dump(self.default_categories, f, indent=2)
        except Exception as e:
            print(f"Error saving default categories: {str(e)}")

    def delete_custom_category(self, category_name: str):
        """Delete a custom category"""
        if isinstance(self.custom_categories,
                      dict) and category_name in self.custom_categories:
            del self.custom_categories[category_name]
            self._save_custom_categories()

            # Update category patterns
            self.category_patterns = self._initialize_category_patterns()
        elif isinstance(self.custom_categories, list):
            # Remove all rules for this category
            self.custom_categories = [
                rule for rule in self.custom_categories
                if rule.get('category') != category_name
            ]
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
            custom_cats = list(
                set([
                    rule.get('category', 'Others')
                    for rule in self.custom_categories
                    if isinstance(rule, dict)
                ]))
        else:
            custom_cats = []

        return default_cats + custom_cats

    def get_default_categories(self) -> Dict[str, Dict]:
        """Get default categories with their keywords and types"""
        # Ensure we always return a dict format
        if isinstance(self.default_categories, dict):
            return self.default_categories.copy()
        else:
            # Convert old list format to new dict format
            converted = {}
            for category in self.default_categories:
                converted[category] = {
                    "keywords": self.categories.get(category, []),
                    "type": "expense"
                }
            return converted

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
                        result[category]['keywords'].append(
                            rule.get('pattern', ''))
            return result

    def get_category_type(self, category_name: str) -> str:
        """Get the type of a specific category"""
        if category_name in self.default_categories:
            return self.default_categories[category_name].get(
                'type', 'expense')
        elif isinstance(self.custom_categories,
                        dict) and category_name in self.custom_categories:
            return self.custom_categories[category_name].get('type', 'expense')
        elif isinstance(self.custom_categories, list):
            # Find first rule with this category
            for rule in self.custom_categories:
                if isinstance(rule,
                              dict) and rule.get('category') == category_name:
                    return rule.get('type', 'expense')
        return 'expense'

    def get_cache_statistics(self) -> Dict:
        """Get statistics about the categorization cache"""
        try:
            cache_size = os.path.getsize(self.cache_file) if os.path.exists(
                self.cache_file) else 0
            cache_size_kb = round(cache_size / 1024, 2)

            return {
                "pattern_count":
                len(self.categorization_cache.get("patterns", {})),
                "correction_count":
                len(self.categorization_cache.get("corrections", {})),
                "fuzzy_match_count":
                len(self.categorization_cache.get("fuzzy_matches", {})),
                "cache_size_kb":
                cache_size_kb,
                "last_updated":
                self.categorization_cache.get("last_updated", "Never")
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

    def analyze_narration_with_ai(self, narration: str) -> dict:
        """Use AI to analyze narration and suggest category with reasoning"""
        if not TOGETHER_API_KEY:
            return {
                "purpose":
                "AI analysis not available - API key not configured",
                "suggested_category": "Others",
                "reasoning":
                "AI analysis not available - API key not configured",
                "confidence": 0
            }

        # Log the narration being analyzed
        print(f"Analyzing narration: '{narration}'")

        # Get all available categories
        all_categories = list(self.get_all_categories())

        # Prepare the prompt with structured response format
        categories_info = "".join([
            f"\n- {cat} ({self.get_category_type(cat)})"
            for cat in all_categories
        ])

        prompt = f"""Analyze this bank transaction and categorize it.

Available categories:{categories_info}

Transaction: "{narration}"

Respond with JSON:
{{
    "purpose": "brief description",
    "suggested_category": "exact category name",
    "reasoning": "why this category",
    "confidence": 85
}}"""

        try:
            client = Together(api_key=TOGETHER_API_KEY)

            response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.1,
                max_tokens=150)

            result_text = response.choices[0].message.content.strip(
            ) if response.choices and response.choices[0].message else ""

            print(f"AI Response: {result_text}")  # Debug log

            if not result_text:
                print("Empty AI response received")
                return {
                    "purpose": "Empty Response",
                    "suggested_category": "Others",
                    "reasoning": "AI returned empty response",
                    "confidence": 0
                }

            # Try to parse JSON response
            try:
                # Clean the response to extract JSON if wrapped in markdown
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split(
                        "```")[0].strip()
                elif "```" in result_text:
                    result_text = result_text.split("```")[1].strip()

                result = json.loads(result_text)

                # Validate and sanitize the response
                if not isinstance(result, dict):
                    raise ValueError("Response is not a JSON object")

                # Ensure required fields exist
                result['purpose'] = result.get('purpose', 'Transaction')
                result['suggested_category'] = result.get(
                    'suggested_category', 'Others')
                result['reasoning'] = result.get('reasoning',
                                                 'AI analysis completed')
                result['confidence'] = max(
                    0, min(100, result.get('confidence', 50)))

                # Validate suggested category is in our list
                if result['suggested_category'] not in all_categories:
                    result['suggested_category'] = 'Others'
                    result[
                        'reasoning'] = f"Original suggestion not in available categories. {result['reasoning']}"

                return result

            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing AI response: {result_text}, Error: {e}")

                # Fallback: try to extract category from text response
                suggested_category = 'Others'
                for category in all_categories:
                    if category.lower() in result_text.lower():
                        suggested_category = category
                        break

                return {
                    "purpose": "AI analysis",
                    "suggested_category": suggested_category,
                    "reasoning":
                    f"Extracted from text response: {result_text[:100]}...",
                    "confidence": 30
                }

        except Exception as e:
            print(f"AI Analysis Error: {e}")
            return {
                "purpose": "Analysis failed",
                "suggested_category": "Others",
                "reasoning": f"AI request failed: {str(e)}",
                "confidence": 0
            }

    def _batch_analyze_with_ai(self, narrations: List[str]) -> List[dict]:
        """Batch analyze multiple narrations with AI for efficiency"""
        if not TOGETHER_API_KEY or not narrations:
            return [{
                "suggested_category": "Others",
                "confidence": 0
            } for _ in narrations]

        results = []
        all_categories = list(self.get_all_categories())

        # Get categories with their types for context
        categories_info = ""
        for cat in all_categories:
            cat_type = self.get_category_type(cat)
            categories_info += f"\n- {cat} ({cat_type})"

        # Process in batches to avoid token limits
        batch_size = 5
        for i in range(0, len(narrations), batch_size):
            batch = narrations[i:i + batch_size]

            # Create batch prompt
            batch_prompt = f"""
            Analyze these bank transaction narrations and categorize each one.

            Available categories with types:{categories_info}

            Transactions:
            """

            for j, narration in enumerate(batch):
                batch_prompt += f"\n{j+1}. \"{narration}\""

            batch_prompt += f"""

            Respond with a JSON array where each object has:
            {{
                "purpose": "brief description in 2-4 words",
                "suggested_category": "exact category name from the list",
                "transaction_type": "income/expense/asset/liability",
                "reasoning": "why this category fits", 
                "confidence": 85
            }}
            """

            try:
                client = Together(api_key=TOGETHER_API_KEY)

                response = client.chat.completions.create(
                    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                    messages=[{
                        "role": "user",
                        "content": batch_prompt
                    }],
                    temperature=0.2,
                    max_tokens=500)

                result_text = response.choices[0].message.content.strip()

                try:
                    batch_results = json.loads(result_text)
                    if isinstance(batch_results, list):
                        for result in batch_results:
                            # Validate suggested category
                            if result.get('suggested_category'
                                          ) not in all_categories:
                                result['suggested_category'] = 'Others'
                        results.extend(batch_results)
                    else:
                        # Single result instead of array
                        if batch_results.get(
                                'suggested_category') not in all_categories:
                            batch_results['suggested_category'] = 'Others'
                        results.extend([batch_results] * len(batch))
                except json.JSONDecodeError:
                    # Fallback for parsing errors
                    results.extend([{
                        "suggested_category": "Others",
                        "confidence": 0
                    }] * len(batch))

            except Exception as e:
                print(f"Batch AI analysis error: {e}")
                results.extend([{
                    "suggested_category": "Others",
                    "confidence": 0
                }] * len(batch))

        return results

    def test_categorization(self, test_narrations=None):
        """Test categorization with sample narrations"""
        if test_narrations is None:
            test_narrations = [
                "tution", "K Singhvi and Associates", "Vegetables"
            ]

        print("=== Categorization Test ===")
        print(
            f"Available custom categories: {list(self.custom_categories.keys()) if isinstance(self.custom_categories, dict) else 'List format'}"
        )

        for narration in test_narrations:
            category, ai_flag, similarity = self._categorize_single_transaction_with_similarity(
                narration)
            print(
                f"'{narration}' -> '{category}' (AI: {ai_flag}, Similarity: {similarity}%)"
            )
        print("=== End Test ===\n")

    def categorize_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Categorize transactions using enhanced pattern matching and AI

        Args:
            df: DataFrame with transaction data

        Returns:
            DataFrame with added 'category' column
        """
        if df.empty:
            return df

        df_copy = df.copy()

        # Initialize columns if they don't exist
        if 'category' not in df_copy.columns:
            df_copy['category'] = 'Others'
        if 'ai_categorized' not in df_copy.columns:
            df_copy['ai_categorized'] = False
        if 'similarity_score' not in df_copy.columns:
            df_copy['similarity_score'] = 0.0

        try:
            # Step 1: Rule-based categorization with similarity scoring
            pending_for_ai = []

            for idx, row in df_copy.iterrows():
                if pd.isna(row['narration']) or row['narration'].strip() == '':
                    continue

                narration_lower = str(row['narration']).lower().strip()

                # Check cache first
                if narration_lower in self.categorization_cache.get(
                        "patterns", {}):
                    cached_result = self.categorization_cache["patterns"][
                        narration_lower]
                    if isinstance(cached_result, str):
                        df_copy.loc[idx, 'category'] = cached_result
                        df_copy.loc[idx, 'similarity_score'] = 100
                        df_copy.loc[idx, 'ai_categorized'] = False
                        continue
                    elif isinstance(cached_result, dict):
                        df_copy.loc[idx, 'category'] = cached_result.get(
                            'category', 'Others')
                        df_copy.loc[idx,
                                    'similarity_score'] = cached_result.get(
                                        'confidence', 0) * 100
                        df_copy.loc[idx, 'ai_categorized'] = True
                        continue

                # Apply rule-based categorization
                category, similarity_score = self._apply_similarity_based_categorization(
                    narration_lower)

                if similarity_score >= 30:  # Lowered threshold to allow more AI usage
                    # High confidence rule match
                    df_copy.loc[idx, 'category'] = category
                    df_copy.loc[idx, 'similarity_score'] = similarity_score
                    df_copy.loc[idx, 'ai_categorized'] = False

                    # Cache the result
                    self.categorization_cache["patterns"][
                        narration_lower] = category
                else:
                    # Low confidence or no match - add to AI pending list
                    pending_for_ai.append((idx, narration_lower))

            # Step 2: Batch AI categorization for pending transactions
            if pending_for_ai and TOGETHER_API_KEY:
                # Prepare batch prompt for AI
                narrations_for_ai = [item[1] for item in pending_for_ai]
                ai_results = self._batch_analyze_with_ai(narrations_for_ai)

                for i, (idx, narration) in enumerate(pending_for_ai):
                    if i < len(ai_results) and ai_results[i]:
                        result = ai_results[i]
                        confidence = result.get('confidence', 0)
                        suggested_category = result.get(
                            'suggested_category', 'Others')

                        if confidence >= 30:  # Lowered threshold for AI
                            # AI has good confidence
                            df_copy.loc[idx, 'category'] = suggested_category
                            df_copy.loc[idx, 'similarity_score'] = confidence
                            df_copy.loc[idx, 'ai_categorized'] = True

                            # Cache the AI result
                            self.categorization_cache["patterns"][
                                narration] = {
                                    "category": suggested_category,
                                    "confidence": confidence / 100.0,
                                    "timestamp": datetime.now().isoformat()
                                }
                        else:
                            # Low AI confidence - put in Suspense
                            df_copy.loc[idx, 'category'] = 'Suspense'
                            df_copy.loc[idx, 'similarity_score'] = confidence
                            df_copy.loc[idx, 'ai_categorized'] = True
                    else:
                        # AI failed or no result - put in Suspense
                        df_copy.loc[idx, 'category'] = 'Suspense'
                        df_copy.loc[idx, 'similarity_score'] = 0
                        df_copy.loc[idx, 'ai_categorized'] = True
            else:
                # No AI available - put all pending in Suspense
                for idx, narration in pending_for_ai:
                    df_copy.loc[idx, 'category'] = 'Suspense'
                    df_copy.loc[idx, 'similarity_score'] = 0
                    df_copy.loc[idx, 'ai_categorized'] = False

            # Save updated cache
            self._save_categorization_cache()

        except Exception as e:
            print(f"Error during categorization: {str(e)}")

        return df_copy
