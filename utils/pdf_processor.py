import streamlit as st
import PyPDF2
import pdfplumber
import io
import re
from typing import Union, BinaryIO, List, Dict

class PDFProcessor:
    """Handles PDF text extraction and transaction parsing"""

    def __init__(self):
        self.extraction_methods = [
            self._extract_with_pdfplumber,
            self._extract_with_pypdf2
        ]

    def extract_text(self, uploaded_file) -> str:
        uploaded_file.seek(0)
        file_content = uploaded_file.read()

        for method in self.extraction_methods:
            try:
                text = method(io.BytesIO(file_content))
                if text.strip():
                    return text
            except Exception as e:
                st.warning(f"Extraction method failed: {str(e)}")
                continue

        raise Exception("Could not extract text from PDF using any available method")

    def _extract_with_pdfplumber(self, file_stream: BinaryIO) -> str:
        text_content = []
        with pdfplumber.open(file_stream) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_content.append(text)
        return '\n'.join(text_content)

    def _extract_with_pypdf2(self, file_stream: BinaryIO) -> str:
        text_content = []
        pdf_reader = PyPDF2.PdfReader(file_stream)
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                text_content.append(text)
        return '\n'.join(text_content)

    def validate_pdf(self, uploaded_file) -> bool:
        try:
            uploaded_file.seek(0)
            file_content = uploaded_file.read()
            if not file_content.startswith(b'%PDF'):
                return False
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            return len(pdf_reader.pages) > 0
        except Exception:
            return False
        finally:
            uploaded_file.seek(0)

    def extract_transactions(self, text: str) -> List[Dict]:
        """
        Parse transactions from extracted text.

        Returns:
            List of transactions with inferred type (income/expense)
        """
        lines = text.split("\n")
        transactions = []

        for line in lines:
            # Regex to detect lines with date, withdrawals, deposits, and balance
            match = re.match(
                r"(\d{2}-\d{2}-\d{4}).*?(-?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)?\s*(-?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(-?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
                line.replace(",", "")
            )
            if match:
                date, withdrawal, deposit, balance = match.groups()
                amount = 0.0
                txn_type = "unknown"

                if withdrawal:
                    amount = -float(withdrawal.replace(",", ""))
                    txn_type = "debit"
                elif deposit:
                    amount = float(deposit.replace(",", ""))
                    txn_type = "credit"

                if amount != 0:
                    transactions.append({
                        'date': date,
                        'narration': line.strip()[:100],  # First 100 chars as narration
                        'amount': amount,
                        'type': txn_type,
                        'source_file': 'PDF'
                    })

        return transactions