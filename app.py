import streamlit as st

# Set page config first (must be the first Streamlit command)
st.set_page_config(
    page_title="Bank Statement Analyzer",
    page_icon="🏦",
    layout="wide"
)

import pandas as pd
import os
from datetime import datetime
import json

# Import utility modules
from utils.pdf_processor import PDFProcessor
fromutils.excel_processor import ExcelProcessor
from utils.transaction_parser import TransactionParser
from utils.ai_categorizer import AICategorizer
from utils.data_manager import DataManager
from utils.chart_generator import ChartGenerator
from utils.audit_logger import AuditLogger
from utils.rules_engine import RulesEngine
from utils.category_manager import CategoryManager



# Initialize components
@st.cache_resource
def init_components():
    """Initialize all utility components"""
    pdf_processor = PDFProcessor()
    excel_processor = ExcelProcessor()
    transaction_parser = TransactionParser()
    ai_categorizer = AICategorizer()
    data_manager = DataManager()
    chart_generator = ChartGenerator()
    audit_logger = AuditLogger()
    rules_engine = RulesEngine()
    category_manager = CategoryManager()
    return (pdf_processor, excel_processor, transaction_parser, ai_categorizer, data_manager, 
            chart_generator, audit_logger, rules_engine, category_manager)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'transactions' not in st.session_state:
        st.session_state.transactions = pd.DataFrame()
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'upload'

def render_upload_view(pdf_proc, excel_proc, trans_parser, ai_cat, data_mgr, audit_log):
    """Render the upload and process view"""
    st.header("📄 Upload Bank Statements")

    # File format selection
    upload_format = st.radio(
        "Choose file format:",
        ["PDF Bank Statements", "Excel/CSV Data"],
        help="Select whether you want to upload PDF bank statements or Excel/CSV files with transaction data"
    )

    if upload_format == "PDF Bank Statements":
        uploaded_files = st.file_uploader(
            "Choose PDF files", 
            type=['pdf'], 
            accept_multiple_files=True,
            help="Upload your bank statement PDFs. Supports most Indian bank formats."
        )
    else:
        uploaded_files = st.file_uploader(
            "Choose Excel/CSV files", 
            type=['xlsx', 'xls', 'csv'], 
            accept_multiple_files=True,
            help="Upload Excel or CSV files with columns: date, narration, withdrawal, deposits"
        )

        # Show sample format
        with st.expander("📋 Expected File Format"):
            st.write("Your Excel/CSV file should have these columns:")
            st.code(excel_proc.get_sample_format(), language="csv")
            st.write("**Column descriptions:**")
            st.write("- **date**: Transaction date (DD/MM/YYYY, DD-MM-YYYY, or YYYY-MM-DD)")
            st.write("- **narration**: Transaction description")
            st.write("- **withdrawal**: Amount debited (leave empty if not applicable)")
            st.write("- **deposits**: Amount credited (leave empty if not applicable)")

    if uploaded_files:
        if st.button("Process Statements", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            all_transactions = []

            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")
                progress_bar.progress((i + 0.5) / len(uploaded_files))

                try:
                    if upload_format == "PDF Bank Statements":
                        # Process PDF files
                        text_content = pdf_proc.extract_text(file)
                        transactions_list = pdf_proc.extract_transactions(text_content)
                        transactions = trans_parser.convert_to_dataframe(transactions_list, file.name)
                    else:
                        # Process Excel/CSV files
                        transactions = excel_proc.extract_transactions(file, file.name)

                    if not transactions.empty:
                        # AI categorization
                        categorized_transactions = ai_cat.categorize_transactions(transactions)
                        all_transactions.append(categorized_transactions)

                        # Log audit trail
                        audit_log.log_upload(file.name, len(transactions))

                        st.success(f"✅ Processed {file.name}: {len(transactions)} transactions found")
                    else:
                        st.warning(f"⚠️ No transactions found in {file.name}")

                except Exception as e:
                    st.error(f"❌ Error processing {file.name}: {str(e)}")

                progress_bar.progress((i + 1) / len(uploaded_files))

            if all_transactions:
                # Combine all transactions
                combined_df = pd.concat(all_transactions, ignore_index=True)
                st.session_state.transactions = combined_df
                data_mgr.save_transactions(combined_df)

                status_text.text("✅ All files processed successfully!")
                st.balloons()

                # Show summary
                st.subheader("📊 Processing Summary")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Transactions", len(combined_df))
                with col2:
                    st.metric("Total Income", f"₹{combined_df[combined_df['amount'] > 0]['amount'].sum():,.2f}")
                with col3:
                    st.metric("Total Expenses", f"₹{abs(combined_df[combined_df['amount'] < 0]['amount'].sum()):,.2f}")

def render_dashboard_view(data_mgr, chart_gen):
    """Render the dashboard view"""
    st.header("📊 Financial Dashboard")

    # Load existing transactions if available
    if st.session_state.transactions.empty:
        existing_transactions = data_mgr.load_transactions()
        if not existing_transactions.empty:
            st.session_state.transactions = existing_transactions

    df = st.session_state.transactions

    if df.empty:
        st.info("No transactions available. Please upload and process bank statements first.")
        return

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)

    total_income = df[df['amount'] > 0]['amount'].sum()
    total_expenses = abs(df[df['amount'] < 0]['amount'].sum())
    net_savings = total_income - total_expenses
    transaction_count = len(df)

    with col1:
        st.metric("Total Income", f"₹{total_income:,.2f}")
    with col2:
        st.metric("Total Expenses", f"₹{total_expenses:,.2f}")
    with col3:
        st.metric("Net Savings", f"₹{net_savings:,.2f}")
    with col4:
        st.metric("Transactions", transaction_count)

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💰 Spending by Category")
        category_chart = chart_gen.create_category_pie_chart(df)
        if category_chart:
            st.plotly_chart(category_chart, use_container_width=True)

    with col2:
        st.subheader("📈 Monthly Trend")
        trend_chart = chart_gen.create_monthly_trend_chart(df)
        if trend_chart:
            st.plotly_chart(trend_chart, use_container_width=True)

    # Suspicious Transactions
    st.subheader("🚨 Suspicious Transactions")
    suspicious_transactions = data_mgr.detect_suspicious_transactions(df)

    if not suspicious_transactions.empty:
        st.dataframe(suspicious_transactions[['date', 'narration', 'amount', 'category', 'suspicion_reason']], 
                    use_container_width=True)
    else:
        st.info("No suspicious transactions detected.")

def render_transaction_details_view(data_mgr, ai_cat, audit_log):
    """Render the transaction details view"""
    st.header("📋 Transaction Details")

    if st.session_state.transactions.empty:
        existing_transactions = data_mgr.load_transactions()
        if not existing_transactions.empty:
            st.session_state.transactions = existing_transactions

    df = st.session_state.transactions

    if df.empty:
        st.info("No transactions available. Please upload and process bank statements first.")
        return

    # Category filter
    categories = ['All'] + sorted(df['category'].unique().tolist())
    selected_category = st.selectbox("Filter by Category", categories)

    # Date range filter
    col1, col2 = st.columns(2)
    with col1:
        try:
                # Try to parse dates with multiple formats
                df['date_parsed'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
                min_date = df['date_parsed'].min()
                if pd.isna(min_date):
                    min_date = pd.Timestamp.now() - pd.Timedelta(days=30)
                start_date = st.date_input("From Date", value=min_date)
            except:
                start_date = st.date_input("From Date", value=pd.Timestamp.now() - pd.Timedelta(days=30))
    with col2:
        try:
                # Try to parse dates with multiple formats
                if 'date_parsed' not in df.columns:
                    df['date_parsed'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
                max_date = df['date_parsed'].max()
                if pd.isna(max_date):
                    max_date = pd.Timestamp.now()
                end_date = st.date_input("To Date", value=max_date)
            except:
                end_date = st.date_input("To Date", value=pd.Timestamp.now())

    # Filter data
    filtered_df = df.copy()
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['category'] == selected_category]

    filtered_df = filtered_df[
        (pd.to_datetime(filtered_df['date'], format='mixed', dayfirst=True, errors='coerce') >= pd.to_datetime(start_date)) &
        (pd.to_datetime(filtered_df['date'], format='mixed', dayfirst=True, errors='coerce') <= pd.to_datetime(end_date))
    ]

    # Transaction editing interface
    st.subheader("Edit Transactions")
    st.info("Click on any row to edit its category. Changes are automatically saved and improve AI learning.")

    # Get available categories
    available_categories = ai_cat.get_all_categories()

    # Display editable transactions
    for idx, row in filtered_df.iterrows():
        # Check if transaction was AI categorized
        ai_label = " 🤖 AI" if row.get('ai_categorized', False) else ""

        with st.expander(f"{row['date']} - {row['narration'][:50]}... - ₹{row['amount']:,.2f}{ai_label}"):
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.text(f"Narration: {row['narration']}")
                st.text(f"Amount: ₹{row['amount']:,.2f}")
                st.text(f"Source: {row['source_file']}")
                if row.get('ai_categorized', False):
                    st.caption("🤖 Categorized by AI")

            with col2:
                current_category = row['category']
                new_category = st.selectbox(
                    "Category",
                    available_categories,
                    index=available_categories.index(current_category) if current_category in available_categories else 0,
                    key=f"cat_{idx}"
                )

            with col3:
                if st.button("Update", key=f"update_{idx}"):
                    if new_category != current_category:
                        # Update transaction
                        st.session_state.transactions.loc[idx, 'category'] = new_category
                        st.session_state.transactions.loc[idx, 'ai_categorized'] = False  # User correction

                        # Update AI learning cache
                        ai_cat.learn_from_correction(str(row['narration']), new_category)

                        # Save changes
                        data_mgr.save_transactions(st.session_state.transactions)

                        # Log audit trail
                        audit_log.log_categorization_change(str(row['narration']), current_category, new_category)

                        st.success(f"Updated category to {new_category}")
                        st.rerun()

def render_category_management_view(ai_cat, audit_log, cat_mgr):
    """Render the category management view"""
    st.header("🏷️ Category Management")

    # Add categorization rules
    st.subheader("Add Categorization Rules")
    with st.form("Add New Categorization Rule"):
        pattern = st.text_input("Rule Pattern (Regex)", help="Example: NACH/.*/DIV")
        category = st.selectbox("Select Category", options=["Salary", "Dividend", "Food", "Investment", "EMI", "Others"])
        submit = st.form_submit_button("➕ Add Rule and Apply")

        if submit and pattern:
            st.session_state.transactions = cat_mgr.add_and_apply_rule(pattern, category, st.session_state.transactions)
            st.success("✅ Rule added and applied to all matching transactions.")

    # Custom category creation
    st.subheader("Create Custom Category")
    col1, col2, col3 = st.columns(3)

    with col1:
        new_category_name = st.text_input("Category Name", placeholder="e.g., Gym Membership")

    with col2:
        new_category_keywords = st.text_input("Keywords (comma-separated)", placeholder="e.g., gym, fitness, workout")

    with col3:
        category_type = st.selectbox("Category Type", ["income", "expense", "asset", "liability"], index=1)

    if st.button("Add Custom Category"):
        if new_category_name and new_category_keywords:
            keywords_list = [kw.strip() for kw in new_category_keywords.split(',')]
            ai_cat.add_custom_category(new_category_name, keywords_list, category_type)
            audit_log.log_category_creation(new_category_name, keywords_list)
            st.success(f"Added custom category: {new_category_name} ({category_type})")
            st.rerun()
        else:
            st.error("Please provide both category name and keywords")

    # Display existing categories
    st.subheader("Existing Categories")

    # Default categories
    st.write("**Default Categories:**")
    default_categories = ai_cat.get_default_categories()
    for category in default_categories:
        category_type = ai_cat.get_category_type(category)
        st.write(f"• {category} ({category_type})")

    # Custom categories
    st.write("**Custom Categories:**")
    custom_categories = ai_cat.get_custom_categories()

    if custom_categories:
        for category, category_data in custom_categories.items():
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

            # Handle both old and new format
            if isinstance(category_data, dict):
                keywords = category_data.get('keywords', [])
                category_type = category_data.get('type', 'expense')
            else:
                keywords = category_data if isinstance(category_data, list) else []
                category_type = 'expense'

            with col1:
                st.write(f"**{category}**")
            with col2:
                st.write(f"Type: {category_type}")
            with col3:
                st.write(f"Keywords: {', '.join(keywords)}")
            with col4:
                if st.button("Delete", key=f"del_{category}"):
                    ai_cat.delete_custom_category(category)
                    audit_log.log_category_deletion(category)
                    st.success(f"Deleted category: {category}")
                    st.rerun()
    else:
        st.info("No custom categories created yet.")

    # Categorization cache management
    st.subheader("AI Learning Cache")
    cache_stats = ai_cat.get_cache_statistics()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cached Patterns", cache_stats.get('pattern_count', 0))
    with col2:
        st.metric("Learning Corrections", cache_stats.get('correction_count', 0))
    with col3:
        st.metric("Cache Size (KB)", cache_stats.get('cache_size_kb', 0))

    if st.button("Clear AI Cache", type="secondary"):
        ai_cat.clear_cache()
        audit_log.log_cache_operation("clear", {"cache_size_before": cache_stats.get('cache_size_kb', 0)})
        st.success("AI cache cleared successfully")
        st.rerun()

def render_audit_trail_view(audit_log):
    """Render the audit trail view"""
    st.header("📜 Audit Trail")

    audit_data = audit_log.get_audit_log()

    if audit_data:
        df_audit = pd.DataFrame(audit_data)

        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            action_filter = st.selectbox("Filter by Action", ["All"] + list(df_audit['action'].unique()))
        with col2:
            date_filter = st.date_input("From Date", value=datetime.now().date())

        # Apply filters
        filtered_audit = df_audit.copy()
        if action_filter != "All":
            filtered_audit = filtered_audit[filtered_audit['action'] == action_filter]

        filtered_audit = filtered_audit[
            pd.to_datetime(filtered_audit['timestamp']).dt.date >= date_filter
        ]

        # Display audit log
        st.dataframe(filtered_audit, use_container_width=True)

        # Export option
        if st.button("Export Audit Log"):
            csv = filtered_audit.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No audit trail data available.")

def render_settings_view(data_mgr, ai_cat, audit_log, rules_engine):
    """Render the settings view"""
    st.header("⚙️ Settings")

    # Rules reapplication
    if st.button("🔁 Reapply Custom Categorization Rules"):
        if not st.session_state.transactions.empty:
            updated_df = rules_engine.apply_rules_to_df(st.session_state.transactions)
            st.session_state.transactions = updated_df
            st.success("Rules reapplied to all transactions!")
        else:
            st.warning("No transactions available to update.")

    # Clear all data
    st.subheader("Reset Data")
    st.warning("⚠️ The following actions cannot be undone!")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Clear Transactions", type="secondary"):
            st.session_state.transactions = pd.DataFrame()
            data_mgr.clear_transactions()
            audit_log.log_data_clear("transactions")
            st.success("All transaction data cleared")

    with col2:
        if st.button("Reset All Data", type="secondary"):
            st.session_state.transactions = pd.DataFrame()
            data_mgr.clear_all_data()
            ai_cat.clear_cache()
            audit_log.clear_log()
            audit_log.log_data_clear("all_data")
            st.success("All data reset successfully")
            st.rerun()

    # App information
    st.subheader("About")
    st.info("""
    **Bank Statement Analyzer v1.0**

    This application provides intelligent analysis of bank statements with AI-powered categorization.
    All data is stored locally and never shared externally.

    Features:
    - PDF transaction extraction
    - AI categorization with permanent learning
    - Custom category creation
    - Interactive dashboards
    - Suspicious transaction detection
    - Complete audit trail
    """)

def main():
    """Main application function"""
    # Initialize session state
    init_session_state()

    # Initialize components
    (pdf_proc, excel_proc, trans_parser, ai_cat, data_mgr, chart_gen, 
     audit_log, rules_engine, cat_mgr) = init_components()

    # App title and description
    st.title("🏦 Bank Statement Analyzer")
    st.markdown("Upload your bank statements and get intelligent transaction categorization with AI learning capabilities")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    view_options = ["Upload & Process", "Dashboard", "Transaction Details", 
                   "Category Management", "Audit Trail", "Settings"]
    selected_view = st.sidebar.selectbox("Select View", view_options)

    # Route to appropriate view
    if selected_view == "Upload & Process":
        render_upload_view(pdf_proc, excel_proc, trans_parser, ai_cat, data_mgr, audit_log)
    elif selected_view == "Dashboard":
        render_dashboard_view(data_mgr, chart_gen)
    elif selected_view == "Transaction Details":
        render_transaction_details_view(data_mgr, ai_cat, audit_log)
    elif selected_view == "Category Management":
        render_category_management_view(ai_cat, audit_log, cat_mgr)
    elif selected_view == "Audit Trail":
        render_audit_trail_view(audit_log)
    elif selected_view == "Settings":
        render_settings_view(data_mgr, ai_cat, audit_log, rules_engine)

    # Footer
    st.markdown("---")
    st.markdown("💡 **Tip:** The AI learns from your corrections and gets better over time. All data is stored locally on your device.")

# Run the app
if __name__ == "__main__":
    main()