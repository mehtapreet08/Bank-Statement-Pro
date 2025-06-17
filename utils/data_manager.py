import pandas as pd
import json
import os
from datetime import datetime
from typing import Optional, Dict, List
import numpy as np

class DataManager:
    """Manages transaction data storage and retrieval"""
    
    def __init__(self, user_data_dir: str = None):
        if user_data_dir:
            self.data_dir = user_data_dir
        else:
            self.data_dir = "data"
        
        self.transactions_file = os.path.join(self.data_dir, "transactions.csv")
        self.metadata_file = os.path.join(self.data_dir, "metadata.json")
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
    
    def save_transactions(self, df: pd.DataFrame):
        """Save transactions to CSV file"""
        try:
            df.to_csv(self.transactions_file, index=False)
            self._update_metadata(len(df))
        except Exception as e:
            raise Exception(f"Failed to save transactions: {str(e)}")
    
    def load_transactions(self) -> pd.DataFrame:
        """Load transactions from CSV file"""
        try:
            if os.path.exists(self.transactions_file):
                df = pd.read_csv(self.transactions_file)
                
                # Ensure required columns exist for backward compatibility
                if 'processing_type' not in df.columns:
                    df['processing_type'] = 'Manual'
                if 'confidence' not in df.columns:
                    df['confidence'] = 0.0
                
                return df
            else:
                return pd.DataFrame(columns=['date', 'narration', 'amount', 'type', 'source_file', 'category', 'processing_type', 'confidence'])
        except Exception as e:
            print(f"Failed to load transactions: {str(e)}")
            return pd.DataFrame(columns=['date', 'narration', 'amount', 'type', 'source_file', 'category', 'processing_type', 'confidence'])
    
    def append_transactions(self, new_df: pd.DataFrame):
        """Append new transactions to existing data"""
        existing_df = self.load_transactions()
        
        if not existing_df.empty:
            # Remove duplicates based on date, narration, and amount
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(
                subset=['date', 'narration', 'amount'], 
                keep='last'
            )
        else:
            combined_df = new_df
        
        self.save_transactions(combined_df)
        return combined_df
    
    def _update_metadata(self, transaction_count: int):
        """Update metadata file with current statistics"""
        metadata = {
            "last_updated": datetime.now().isoformat(),
            "transaction_count": transaction_count,
            "file_size_bytes": os.path.getsize(self.transactions_file) if os.path.exists(self.transactions_file) else 0
        }
        
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception:
            pass  # Metadata update failure shouldn't break the main functionality
    
    def get_metadata(self) -> Dict:
        """Get current metadata"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        
        return {
            "last_updated": "Never",
            "transaction_count": 0,
            "file_size_bytes": 0
        }
    
    def detect_suspicious_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect potentially suspicious transactions
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with suspicious transactions and reasons
        """
        if df.empty:
            return pd.DataFrame()
        
        suspicious_transactions = []
        df_copy = df.copy()
        df_copy['amount_abs'] = df_copy['amount'].abs()
        
        # Filter out already approved transactions
        if 'approved' in df_copy.columns:
            df_copy = df_copy[df_copy['approved'] != True]
        
        # 1. Repeated identical amounts on same day
        same_day_amounts = df_copy.groupby(['date', 'amount_abs']).size()
        repeated_amounts = same_day_amounts[same_day_amounts > 1].index
        
        for date, amount in repeated_amounts:
            matching_transactions = df_copy[
                (df_copy['date'] == date) & 
                (df_copy['amount_abs'] == amount)
            ]
            
            for _, transaction in matching_transactions.iterrows():
                suspicious_transactions.append({
                    **transaction.to_dict(),
                    'suspicion_reason': f'Repeated amount ₹{amount:,.2f} on same day'
                })
        
        # 2. Round number transactions above threshold
        round_number_threshold = 10000
        round_numbers = df_copy[
            (df_copy['amount_abs'] % 1000 == 0) & 
            (df_copy['amount_abs'] >= round_number_threshold)
        ]
        
        for _, transaction in round_numbers.iterrows():
            suspicious_transactions.append({
                **transaction.to_dict(),
                'suspicion_reason': f'Large round number transaction: ₹{transaction["amount_abs"]:,.2f}'
            })
        
        # 3. High frequency transactions (more than 5 per day)
        daily_transaction_count = df_copy.groupby(['date']).size()
        high_frequency_dates = daily_transaction_count[daily_transaction_count > 5].index
        
        for date in high_frequency_dates:
            date_transactions = df_copy[df_copy['date'] == date]
            for _, transaction in date_transactions.iterrows():
                suspicious_transactions.append({
                    **transaction.to_dict(),
                    'suspicion_reason': f'High frequency day: {daily_transaction_count[date]} transactions'
                })
        
        # 4. Large transactions (above 95th percentile)
        if len(df_copy) > 10:  # Only if we have enough data
            large_transaction_threshold = df_copy['amount_abs'].quantile(0.95)
            large_transactions = df_copy[df_copy['amount_abs'] > large_transaction_threshold]
            
            for _, transaction in large_transactions.iterrows():
                suspicious_transactions.append({
                    **transaction.to_dict(),
                    'suspicion_reason': f'Large transaction: ₹{transaction["amount_abs"]:,.2f} (top 5%)'
                })
        
        # 5. Cash deposits/withdrawals above threshold
        cash_threshold = 50000
        cash_keywords = ['cash', 'atm', 'withdrawal', 'deposit']
        
        for _, transaction in df_copy.iterrows():
            if (transaction['amount_abs'] > cash_threshold and 
                any(keyword in transaction['narration'].lower() for keyword in cash_keywords)):
                suspicious_transactions.append({
                    **transaction.to_dict(),
                    'suspicion_reason': f'Large cash transaction: ₹{transaction["amount_abs"]:,.2f}'
                })
        
        # Convert to DataFrame and remove duplicates
        if suspicious_transactions:
            suspicious_df = pd.DataFrame(suspicious_transactions)
            # Remove duplicates based on transaction details, keep the first reason
            suspicious_df = suspicious_df.drop_duplicates(
                subset=['date', 'narration', 'amount'], 
                keep='first'
            )
            return suspicious_df
        else:
            return pd.DataFrame()
    
    def get_transaction_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics for transactions"""
        if df.empty:
            return {
                "total_transactions": 0,
                "total_income": 0,
                "total_expenses": 0,
                "net_savings": 0,
                "average_transaction": 0,
                "date_range": {"start": None, "end": None}
            }
        
        income_transactions = df[df['amount'] > 0]
        expense_transactions = df[df['amount'] < 0]
        
        total_income = income_transactions['amount'].sum() if not income_transactions.empty else 0
        total_expenses = abs(expense_transactions['amount'].sum()) if not expense_transactions.empty else 0
        
        return {
            "total_transactions": len(df),
            "total_income": total_income,
            "total_expenses": total_expenses,
            "net_savings": total_income - total_expenses,
            "average_transaction": df['amount'].mean(),
            "date_range": {
                "start": df['date'].min(),
                "end": df['date'].max()
            }
        }
    
    def get_category_breakdown(self, df: pd.DataFrame) -> Dict:
        """Get breakdown of transactions by category"""
        if df.empty or 'category' not in df.columns:
            return {}
        
        # Separate income and expenses
        income_df = df[df['amount'] > 0]
        expense_df = df[df['amount'] < 0]
        
        income_breakdown = income_df.groupby('category')['amount'].sum().to_dict()
        expense_breakdown = expense_df.groupby('category')['amount'].apply(lambda x: abs(x.sum())).to_dict()
        
        return {
            "income": income_breakdown,
            "expenses": expense_breakdown
        }
    
    def export_to_csv(self, df: pd.DataFrame, filename: str) -> str:
        """Export DataFrame to CSV and return the content"""
        try:
            csv_content = df.to_csv(index=False)
            return csv_content
        except Exception as e:
            raise Exception(f"Failed to export to CSV: {str(e)}")
    
    def clear_transactions(self):
        """Clear all transaction data"""
        try:
            if os.path.exists(self.transactions_file):
                os.remove(self.transactions_file)
            if os.path.exists(self.metadata_file):
                os.remove(self.metadata_file)
        except Exception as e:
            print(f"Error clearing transactions: {str(e)}")
    
    def clear_all_data(self):
        """Clear all data files"""
        try:
            # List all files in data directory
            if os.path.exists(self.data_dir):
                for filename in os.listdir(self.data_dir):
                    file_path = os.path.join(self.data_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
        except Exception as e:
            print(f"Error clearing all data: {str(e)}")
    
    def backup_data(self) -> str:
        """Create a backup of all data as JSON string"""
        try:
            backup_data = {
                "timestamp": datetime.now().isoformat(),
                "transactions": self.load_transactions().to_dict('records'),
                "metadata": self.get_metadata()
            }
            return json.dumps(backup_data, indent=2)
        except Exception as e:
            raise Exception(f"Failed to create backup: {str(e)}")
    
    def restore_data(self, backup_json: str) -> bool:
        """Restore data from backup JSON string"""
        try:
            backup_data = json.loads(backup_json)
            
            # Restore transactions
            if 'transactions' in backup_data:
                df = pd.DataFrame(backup_data['transactions'])
                self.save_transactions(df)
            
            return True
        except Exception as e:
            print(f"Failed to restore data: {str(e)}")
            return False
