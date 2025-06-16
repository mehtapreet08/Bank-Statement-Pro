
import pandas as pd
import io
from datetime import datetime
from typing import Optional

class ExcelProcessor:
    """Processes Excel/CSV files with transaction data"""
    
    def __init__(self):
        self.required_columns = ['date', 'narration', 'withdrawal', 'deposits']
    
    def validate_file(self, uploaded_file) -> bool:
        """Validate if the uploaded file is a valid Excel/CSV file"""
        try:
            file_extension = uploaded_file.name.lower().split('.')[-1]
            return file_extension in ['xlsx', 'xls', 'csv']
        except Exception:
            return False
    
    def extract_transactions(self, uploaded_file, source_file: str) -> pd.DataFrame:
        """
        Extract transactions from Excel/CSV file
        
        Expected format:
        - date: Transaction date
        - narration: Transaction description
        - withdrawal: Debit amount (negative)
        - deposits: Credit amount (positive)
        
        Returns:
            DataFrame with columns: date, narration, amount, type, source_file
        """
        try:
            # Read file based on extension
            file_extension = uploaded_file.name.lower().split('.')[-1]
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            else:  # xlsx or xls
                df = pd.read_excel(uploaded_file)
            
            # Validate required columns
            missing_columns = [col for col in self.required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Process transactions
            transactions = []
            
            for _, row in df.iterrows():
                # Skip empty rows
                if pd.isna(row['date']) or (pd.isna(row['withdrawal']) and pd.isna(row['deposits'])):
                    continue
                
                # Parse date
                try:
                    if isinstance(row['date'], str):
                        parsed_date = self._parse_date(row['date'])
                    else:
                        parsed_date = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
                except:
                    continue  # Skip rows with invalid dates
                
                # Get narration
                narration = str(row['narration']) if pd.notna(row['narration']) else 'Transaction'
                
                # Determine amount and type
                withdrawal = float(row['withdrawal']) if pd.notna(row['withdrawal']) and row['withdrawal'] != 0 else 0
                deposit = float(row['deposits']) if pd.notna(row['deposits']) and row['deposits'] != 0 else 0
                
                # Create transaction entry
                if withdrawal > 0:
                    # Debit transaction
                    transactions.append({
                        'date': parsed_date,
                        'narration': narration,
                        'amount': -abs(withdrawal),  # Make withdrawal negative
                        'type': 'debit',
                        'source_file': source_file
                    })
                
                if deposit > 0:
                    # Credit transaction
                    transactions.append({
                        'date': parsed_date,
                        'narration': narration,
                        'amount': abs(deposit),  # Make deposit positive
                        'type': 'credit',
                        'source_file': source_file
                    })
            
            # Convert to DataFrame
            if transactions:
                result_df = pd.DataFrame(transactions)
                result_df = self._clean_and_validate_transactions(result_df)
                return result_df
            else:
                return pd.DataFrame(columns=['date', 'narration', 'amount', 'type', 'source_file'])
                
        except Exception as e:
            raise Exception(f"Failed to process Excel/CSV file: {str(e)}")
    
    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse various date formats to standard format"""
        date_formats = [
            '%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y', '%d/%m/%y', '%d-%m-%y',
            '%m/%d/%Y', '%m-%d-%Y', '%m/%d/%y', '%m-%d-%y',
            '%d %b %Y', '%d %B %Y', '%b %d, %Y', '%B %d, %Y'
        ]
        
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(str(date_str).strip(), fmt)
                return parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return None
    
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
    
    def get_sample_format(self) -> str:
        """Return a sample CSV format for user reference"""
        return """date,narration,withdrawal,deposits
2024-01-15,Salary Credit,,50000
2024-01-16,ATM Withdrawal,500,
2024-01-17,Online Purchase,1200,
2024-01-18,Dividend Credit,,250
2024-01-19,UPI Transfer,300,"""
