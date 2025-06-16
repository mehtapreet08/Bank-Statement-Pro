import pandas as pd
import re
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np

class TransactionParser:
    """Parses bank statement text to extract transaction data"""
    
    def __init__(self):
        # Common Indian bank transaction patterns
        self.date_patterns = [
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',  # DD/MM/YYYY or DD-MM-YYYY
            r'\b(\d{2,4}[-/]\d{1,2}[-/]\d{1,2})\b',  # YYYY/MM/DD or YYYY-MM-DD
            r'\b(\d{1,2}\s+\w{3}\s+\d{2,4})\b',      # DD MMM YYYY
            r'\b(\w{3}\s+\d{1,2},?\s+\d{2,4})\b'     # MMM DD, YYYY
        ]
        
        # Amount patterns (Indian format with commas)
        self.amount_patterns = [
            r'(?:Rs\.?|₹)\s*([0-9,]+\.?\d*)',         # Rs. or ₹ prefix
            r'([0-9,]+\.\d{2})\s*(?:Cr|Dr|CR|DR)',   # Amount with Cr/Dr suffix
            r'\b([0-9,]+\.\d{2})\b',                  # Simple decimal amount
            r'\b([0-9,]+)\b(?=\s*(?:Cr|Dr|CR|DR))'   # Whole number with Cr/Dr
        ]
        
        # Credit/Debit indicators
        self.credit_indicators = ['cr', 'credit', 'deposit', '+', 'salary', 'refund']
        self.debit_indicators = ['dr', 'debit', 'withdrawal', '-', 'payment', 'transfer']
        
        # Bank-specific patterns
        self.bank_patterns = {
            'sbi': {
                'transaction_line': r'(\d{2}/\d{2}/\d{4})\s+(.+?)\s+([0-9,]+\.\d{2})\s*(Cr|Dr)',
                'balance_line': r'Balance\s*:?\s*([0-9,]+\.\d{2})'
            },
            'hdfc': {
                'transaction_line': r'(\d{2}-\d{2}-\d{4})\s+(.+?)\s+([0-9,]+\.\d{2})\s*(CR|DR)',
                'balance_line': r'Balance\s*([0-9,]+\.\d{2})'
            },
            'icici': {
                'transaction_line': r'(\d{2}/\d{2}/\d{4})\s+(.+?)\s+([0-9,]+\.\d{2})\s*(CR|DR)',
                'balance_line': r'Available Balance\s*([0-9,]+\.\d{2})'
            },
            'axis': {
                'transaction_line': r'(\d{2}-\w{3}-\d{4})\s+(.+?)\s+([0-9,]+\.\d{2})\s*(CR|DR)',
                'balance_line': r'Balance\s*([0-9,]+\.\d{2})'
            }
        }
    
    def parse_transactions(self, text_content: str, source_file: str) -> pd.DataFrame:
        """
        Parse bank statement text to extract transactions
        
        Args:
            text_content: Raw text from PDF
            source_file: Name of the source file
            
        Returns:
            DataFrame with columns: date, narration, amount, type, source_file
        """
        transactions = []
        
        # Try bank-specific patterns first
        for bank, patterns in self.bank_patterns.items():
            bank_transactions = self._parse_with_bank_pattern(text_content, patterns, source_file)
            if bank_transactions:
                transactions.extend(bank_transactions)
                break
        
        # If no bank-specific pattern worked, use generic parsing
        if not transactions:
            transactions = self._parse_generic(text_content, source_file)
        
        # Convert to DataFrame
        if transactions:
            df = pd.DataFrame(transactions)
            df = self._clean_and_validate_transactions(df)
            return df
        else:
            return pd.DataFrame(columns=['date', 'narration', 'amount', 'type', 'source_file'])
    
    def _parse_with_bank_pattern(self, text: str, patterns: Dict, source_file: str) -> List[Dict]:
        """Parse using bank-specific regex patterns"""
        transactions = []
        
        transaction_pattern = patterns.get('transaction_line')
        if not transaction_pattern:
            return transactions
        
        lines = text.split('\n')
        
        for line in lines:
            match = re.search(transaction_pattern, line, re.IGNORECASE)
            if match:
                try:
                    date_str = match.group(1)
                    narration = match.group(2).strip()
                    amount_str = match.group(3).replace(',', '')
                    cr_dr = match.group(4).upper()
                    
                    # Parse date
                    parsed_date = self._parse_date(date_str)
                    if not parsed_date:
                        continue
                    
                    # Parse amount
                    amount = float(amount_str)
                    if cr_dr in ['DR', 'DEBIT']:
                        amount = -amount
                    
                    transactions.append({
                        'date': parsed_date,
                        'narration': narration,
                        'amount': amount,
                        'type': 'credit' if amount > 0 else 'debit',
                        'source_file': source_file
                    })
                    
                except (ValueError, IndexError):
                    continue
        
        return transactions
    
    def _parse_generic(self, text: str, source_file: str) -> List[Dict]:
        """Generic parsing when bank-specific patterns fail"""
        transactions = []
        lines = text.split('\n')
        
        for line in lines:
            # Skip empty lines and headers
            line = line.strip()
            if not line or len(line) < 10:
                continue
                
            # Look for date patterns
            date_match = None
            for pattern in self.date_patterns:
                date_match = re.search(pattern, line)
                if date_match:
                    break
            
            if not date_match:
                continue
            
            # Look for amount patterns
            amount_matches = []
            for pattern in self.amount_patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                amount_matches.extend(matches)
            
            if not amount_matches:
                continue
            
            # Parse date
            parsed_date = self._parse_date(date_match.group(1))
            if not parsed_date:
                continue
            
            # Extract narration (text between date and amount)
            narration = self._extract_narration(line, date_match.group(1), amount_matches[0])
            
            # Parse amount and determine type
            try:
                amount_str = amount_matches[0].replace(',', '')
                amount = float(amount_str)
                
                # Determine if credit or debit based on context
                transaction_type = self._determine_transaction_type(line, amount)
                if transaction_type == 'debit' and amount > 0:
                    amount = -amount
                
                transactions.append({
                    'date': parsed_date,
                    'narration': narration,
                    'amount': amount,
                    'type': 'credit' if amount > 0 else 'debit',
                    'source_file': source_file
                })
                
            except ValueError:
                continue
        
        return transactions
    
    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse various date formats to standard format"""
        date_formats = [
            '%d/%m/%Y', '%d-%m-%Y', '%d/%m/%y', '%d-%m-%y',
            '%Y/%m/%d', '%Y-%m-%d',
            '%d %b %Y', '%d %B %Y',
            '%b %d, %Y', '%B %d, %Y'
        ]
        
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str.strip(), fmt)
                return parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return None
    
    def _extract_narration(self, line: str, date_str: str, amount_str: str) -> str:
        """Extract transaction narration from the line"""
        # Remove date and amount from line to get narration
        narration = line.replace(date_str, '').replace(amount_str, '')
        
        # Clean up narration
        narration = re.sub(r'[Cc][Rr]|[Dd][Rr]|Rs\.?|₹', '', narration)
        narration = re.sub(r'\s+', ' ', narration).strip()
        
        return narration if narration else 'Transaction'
    
    def _determine_transaction_type(self, line: str, amount: float) -> str:
        """Determine if transaction is credit or debit based on context"""
        line_lower = line.lower()
        
        # Check for explicit table headers and indicators
        if any(word in line_lower for word in ['deposit', 'credit', 'cr', 'credited', 'salary', 'refund', 'interest', 'dividend']):
            return 'credit'
        elif any(word in line_lower for word in ['withdrawal', 'debit', 'dr', 'debited', 'payment', 'transfer', 'emi']):
            return 'debit'
        
        # Check for explicit indicators from patterns
        for indicator in self.credit_indicators:
            if indicator in line_lower:
                return 'credit'
        
        for indicator in self.debit_indicators:
            if indicator in line_lower:
                return 'debit'
        
        # Enhanced logic based on common patterns
        if any(word in line_lower for word in ['salary', 'wages', 'income', 'bonus', 'commission']):
            return 'credit'
        elif any(word in line_lower for word in ['purchase', 'buy', 'paid', 'bill', 'fee', 'charges']):
            return 'debit'
        
        # If unable to determine, assume positive amounts are credits
        return 'credit' if amount > 0 else 'debit'
    
    def _clean_and_validate_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the parsed transactions"""
        if df.empty:
            return df
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['date', 'narration', 'amount'])
        
        # Sort by date
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
        # Clean narration
        df['narration'] = df['narration'].str.strip()
        df['narration'] = df['narration'].str.replace(r'\s+', ' ', regex=True)
        
        # Validate amounts
        df = df[df['amount'] != 0]  # Remove zero amounts
        df = df[pd.notnull(df['amount'])]  # Remove null amounts
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def convert_to_dataframe(self, txn_list, source_file):
        for txn in txn_list:
            txn["narration"] = txn["raw"]  # use raw line as narration
            txn["source_file"] = source_file
        return pd.DataFrame(txn_list)

