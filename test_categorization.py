
#!/usr/bin/env python3

from utils.ai_categorizer import AICategorizer

def main():
    print("Testing categorization...")
    
    # Initialize the categorizer
    ai_cat = AICategorizer()
    
    # Test specific narrations from your data
    test_cases = [
        "tution",
        "K Singhvi and Associates", 
        "Vegetables",
        "Nach/*"
    ]
    
    # Run the test
    ai_cat.test_categorization(test_cases)
    
    # Also test the similarity calculation directly
    print("=== Direct Similarity Tests ===")
    for narration in test_cases:
        for category, category_data in ai_cat.custom_categories.items():
            if isinstance(category_data, dict):
                keywords = category_data.get('keywords', [])
                for keyword in keywords:
                    similarity = ai_cat._calculate_similarity(narration.lower(), str(keyword).lower())
                    if similarity > 0:
                        print(f"'{narration}' vs '{keyword}' ({category}) = {similarity}%")

if __name__ == "__main__":
    main()
