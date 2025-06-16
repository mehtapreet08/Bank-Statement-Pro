# Bank Statement Analyzer

## Overview

This is a Python-based bank statement analyzer built with Streamlit that provides intelligent transaction categorization using AI learning capabilities. The application processes PDF bank statements, extracts transaction data, and automatically categorizes transactions with the ability to learn from user corrections over time.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application
- **Layout**: Multi-view interface with sidebar navigation
- **Components**: File upload, dashboard, transaction details, category management, audit trail, and settings
- **Visualization**: Plotly charts for financial data visualization
- **Port**: Configured to run on port 5000

### Backend Architecture
- **Language**: Python 3.11
- **Architecture Pattern**: Modular utility-based architecture
- **Core Modules**:
  - PDF processing and text extraction
  - Transaction parsing with bank-specific patterns
  - AI-powered categorization with learning cache
  - Data management and persistence
  - Chart generation and visualization
  - Audit logging for all activities

### Data Storage Solutions
- **Primary Storage**: CSV files for transaction data
- **Configuration Storage**: JSON files for categories, cache, and metadata
- **File Structure**:
  - `data/transactions.csv` - Main transaction database
  - `data/categories.json` - Default category patterns
  - `data/custom_categories.json` - User-defined categories
  - `data/categorization_cache.json` - AI learning cache
  - `data/audit_log.json` - Activity audit trail
  - `data/metadata.json` - Application metadata

## Key Components

### PDF Processing (`utils/pdf_processor.py`)
- **Purpose**: Extract text from bank statement PDFs
- **Libraries**: pdfplumber (primary), PyPDF2 (fallback)
- **Features**: Multiple extraction methods with fallback support

### Transaction Parser (`utils/transaction_parser.py`)
- **Purpose**: Parse extracted text to identify transactions
- **Features**: 
  - Support for multiple Indian banks (SBI, HDFC, ICICI, Axis)
  - Flexible date and amount pattern recognition
  - Credit/debit classification

### AI Categorizer (`utils/ai_categorizer.py`)
- **Purpose**: Intelligent transaction categorization with learning
- **Features**:
  - Fuzzy string matching using fuzzywuzzy
  - Permanent learning cache for improved accuracy
  - Custom category support
  - Pattern-based categorization for Indian financial context

### Data Manager (`utils/data_manager.py`)
- **Purpose**: Handle data persistence and retrieval
- **Features**: CSV-based storage with duplicate detection

### Chart Generator (`utils/chart_generator.py`)
- **Purpose**: Create interactive financial visualizations
- **Library**: Plotly for interactive charts
- **Features**: Category pie charts, trend analysis

### Audit Logger (`utils/audit_logger.py`)
- **Purpose**: Track all application activities
- **Features**: JSON-based logging with timestamps

## Data Flow

1. **File Upload**: User uploads PDF bank statement
2. **Text Extraction**: PDF processor extracts text using multiple methods
3. **Transaction Parsing**: Parser identifies and structures transaction data
4. **AI Categorization**: Categorizer assigns categories using patterns and learning
5. **Data Storage**: Transactions saved to CSV with metadata
6. **Visualization**: Charts generated for dashboard display
7. **Audit Logging**: All activities logged for transparency

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **pdfplumber**: PDF text extraction (primary)
- **pypdf2**: PDF processing (fallback)
- **plotly**: Interactive visualization
- **fuzzywuzzy**: Fuzzy string matching for categorization
- **python-levenshtein**: String similarity optimization

### System Dependencies
- **Python**: 3.11+
- **Nix**: stable-24_05 channel
- **Locale**: glibcLocales for internationalization

## Deployment Strategy

### Platform
- **Target**: Replit autoscale deployment
- **Runtime**: Python 3.11 with Nix package management
- **Port Configuration**: 5000 (configured in both .replit and streamlit config)

### Deployment Configuration
- **Run Command**: `streamlit run app.py --server.port 5000`
- **Mode**: Parallel workflow execution
- **Health Check**: Port-based readiness check

### Scalability Considerations
- File-based storage suitable for individual/small business use
- Modular architecture allows easy migration to database storage
- Caching mechanisms reduce processing overhead

## Changelog

```
Changelog:
- June 16, 2025. Initial setup
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```