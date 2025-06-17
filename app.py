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
from utils.excel_processor import ExcelProcessor
from utils.transaction_parser import TransactionParser
from utils.ai_categorizer import AICategorizer
from utils.data_manager import DataManager
from utils.chart_generator import ChartGenerator
from utils.audit_logger import AuditLogger
from utils.rules_engine import RulesEngine
from utils.auth_manager import AuthManager





# Initialize components
@st.cache_resource
def init_components(user_data_dir=None):
    """Initialize all utility components"""
    pdf_processor = PDFProcessor()
    excel_processor = ExcelProcessor()
    transaction_parser = TransactionParser()
    ai_categorizer = AICategorizer(user_data_dir)
    data_manager = DataManager(user_data_dir)
    chart_generator = ChartGenerator()
    audit_logger = AuditLogger()
    rules_engine = RulesEngine()
    return (pdf_processor, excel_processor, transaction_parser, ai_categorizer, data_manager, 
            chart_generator, audit_logger, rules_engine)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'transactions' not in st.session_state:
        st.session_state.transactions = pd.DataFrame()
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'Upload & Process'
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'user_data_dir' not in st.session_state:
        st.session_state.user_data_dir = None

def render_login():
    """Render login/register form"""
    auth_manager = AuthManager()

    st.title("🏦 Bank Statement Analyzer - Login")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_btn = st.form_submit_button("Login")

            if login_btn:
                if auth_manager.authenticate_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.user_data_dir = auth_manager.get_user_data_dir(username)
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("Choose Username")
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            register_btn = st.form_submit_button("Register")

            if register_btn:
                if new_password != confirm_password:
                    st.error("Passwords don't match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                elif auth_manager.register_user(new_username, new_password):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Username already exists")

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

def render_dashboard_view(data_mgr, chart_gen, ai_cat, audit_log):
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

    # Filter out assets and liabilities from income/expense calculations
    asset_categories = [cat for cat in ai_cat.get_all_categories() if ai_cat.get_category_type(cat) == 'asset']
    liability_categories = [cat for cat in ai_cat.get_all_categories() if ai_cat.get_category_type(cat) == 'liability']

    # Filter for only income/expense transactions
    income_expense_df = df[~df['category'].isin(asset_categories + liability_categories)]

    total_income = income_expense_df[income_expense_df['amount'] > 0]['amount'].sum()
    total_expenses = abs(income_expense_df[income_expense_df['amount'] < 0]['amount'].sum())
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
        st.subheader("💰 Income & Spending by Category")
        # Create combined chart for both income and expenses
        category_chart = chart_gen.create_category_pie_chart(income_expense_df)
        if category_chart:
            st.plotly_chart(category_chart, use_container_width=True, key="category_pie")

    with col2:
        st.subheader("📈 Monthly Trend")
        trend_chart = chart_gen.create_monthly_trend_chart(income_expense_df)
        if trend_chart:
            st.plotly_chart(trend_chart, use_container_width=True)

    # Suspicious Transactions - Editable Table Format with Approval
    st.subheader("🚨 Suspicious Transactions")
    suspicious_transactions = data_mgr.detect_suspicious_transactions(df)

    if not suspicious_transactions.empty:
        st.write("Edit categories for suspicious transactions or approve them below:")
        available_categories = ai_cat.get_all_categories()

        # Prepare display dataframe for suspicious transactions
        susp_display_df = suspicious_transactions.copy()
        susp_display_df['Amount (₹)'] = susp_display_df['amount'].apply(lambda x: f"₹{x:,.2f}" if x >= 0 else f"(₹{abs(x):,.2f})")
        susp_display_df['Short Narration'] = susp_display_df['narration'].apply(lambda x: x[:50] + "..." if len(x) > 50 else x)
        susp_display_df['Approve'] = False

        # Create editable dataframe for suspicious transactions
        edited_susp_df = st.data_editor(
            susp_display_df[['Approve', 'date', 'Short Narration', 'Amount (₹)', 'category', 'suspicion_reason']],
            column_config={
                "Approve": st.column_config.CheckboxColumn("Approve"),
                "date": st.column_config.TextColumn("Date"),
                "Short Narration": st.column_config.TextColumn("Narration", width="large"),
                "Amount (₹)": st.column_config.TextColumn("Amount"),
                "category": st.column_config.SelectboxColumn("Category", options=available_categories),
                "suspicion_reason": st.column_config.TextColumn("Suspicion Reason", disabled=True)
            },
            use_container_width=True,
            hide_index=True,
            key="suspicious_editor"
        )

        col1, col2 = st.columns(2)

        with col1:
            # Apply changes to suspicious transactions
            if st.button("💾 Apply Changes to Suspicious Transactions"):
                changes_made = 0
                for i, (edited_row, original_row) in enumerate(zip(edited_susp_df.itertuples(), suspicious_transactions.itertuples())):
                    if edited_row.category != original_row.category:
                        # Find and update in main dataframe
                        mask = (st.session_state.transactions['date'] == original_row.date) & \
                               (st.session_state.transactions['narration'] == original_row.narration) & \
                               (st.session_state.transactions['amount'] == original_row.amount)

                        st.session_state.transactions.loc[mask, 'category'] = edited_row.category
                        audit_log.log_categorization_change(str(original_row.narration), original_row.category, edited_row.category)
                        changes_made += 1

                if changes_made > 0:
                    data_mgr.save_transactions(st.session_state.transactions)
                    st.success(f"✅ Updated {changes_made} suspicious transactions!")
                    st.rerun()
                else:
                    st.info("No changes detected.")

        with col2:
            # Approve suspicious transactions
            if st.button("✅ Approve Selected Transactions"):
                approved_count = 0
                approved_mask = edited_susp_df['Approve'] == True

                if approved_mask.any():
                    # Mark approved transactions as no longer suspicious
                    for i, (edited_row, original_row) in enumerate(zip(edited_susp_df.itertuples(), suspicious_transactions.itertuples())):
                        if edited_row.Approve:
                            # Find and mark as approved in main dataframe
                            mask = (st.session_state.transactions['date'] == original_row.date) & \
                                   (st.session_state.transactions['narration'] == original_row.narration) & \
                                   (st.session_state.transactions['amount'] == original_row.amount)

                            # Add approved flag to avoid future suspicious detection
                            if 'approved' not in st.session_state.transactions.columns:
                                st.session_state.transactions['approved'] = False
                            st.session_state.transactions.loc[mask, 'approved'] = True
                            approved_count += 1

                    if approved_count > 0:
                        data_mgr.save_transactions(st.session_state.transactions)
                        st.success(f"✅ Approved {approved_count} transactions!")
                        st.rerun()
                else:
                    st.warning("No transactions selected for approval.")
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

    # Initialize selection state
    if 'selected_transactions' not in st.session_state:
        st.session_state.selected_transactions = []

    # Search functionality
    search_term = st.text_input("🔍 Search transactions (narration)", placeholder="Type to search...")

    # Category filter
    categories = ['All'] + sorted(df['category'].unique().tolist())

    # Check if a category was selected from the pie chart
    default_category = 'All'
    if hasattr(st.session_state, 'selected_category_filter') and st.session_state.selected_category_filter in categories:
        default_category = st.session_state.selected_category_filter
        # Clear the filter after using it
        del st.session_state.selected_category_filter

    selected_category = st.selectbox("Filter by Category", categories, 
                                   index=categories.index(default_category) if default_category in categories else 0)

    # Filters and sorting
    col1, col2, col3, col4 = st.columns(4)

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

    with col3:
        transaction_type_filter = st.selectbox("Transaction Type", ["All", "Income", "Expense", "Assets", "Liabilities"])

    with col4:
        sort_by = st.selectbox("Sort by", ["Date (Latest)", "Date (Oldest)", "Amount (High to Low)", "Amount (Low to High)", "Category"])

    # Filter data
    filtered_df = df.copy()

    # Apply search filter
    if search_term:
        filtered_df = filtered_df[filtered_df['narration'].str.contains(search_term, case=False, na=False)]

    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['category'] == selected_category]

    # Apply transaction type filter
    if transaction_type_filter == "Income":
        filtered_df = filtered_df[filtered_df['amount'] > 0]
    elif transaction_type_filter == "Expense":
        filtered_df = filtered_df[filtered_df['amount'] < 0]
    elif transaction_type_filter == "Assets":
        asset_categories = [cat for cat in ai_cat.get_all_categories() if ai_cat.get_category_type(cat) == 'asset']
        filtered_df = filtered_df[filtered_df['category'].isin(asset_categories)]
    elif transaction_type_filter == "Liabilities":
        liability_categories = [cat for cat in ai_cat.get_all_categories() if ai_cat.get_category_type(cat) == 'liability']
        filtered_df = filtered_df[filtered_df['category'].isin(liability_categories)]

    # Apply date filter
    try:
        filtered_df = filtered_df[
            (pd.to_datetime(filtered_df['date'], format='mixed', dayfirst=True, errors='coerce') >= pd.to_datetime(start_date)) &
            (pd.to_datetime(filtered_df['date'], format='mixed', dayfirst=True, errors='coerce') <= pd.to_datetime(end_date))
        ]
    except:
        pass  # If date filtering fails, continue without it

    # Apply sorting
    try:
        if sort_by == "Date (Latest)":
            filtered_df = filtered_df.sort_values('date_parsed', ascending=False, na_position='last')
        elif sort_by == "Date (Oldest)":
            filtered_df = filtered_df.sort_values('date_parsed', ascending=True, na_position='last')
        elif sort_by == "Amount (High to Low)":
            filtered_df = filtered_df.sort_values('amount', ascending=False)
        elif sort_by == "Amount (Low to High)":
            filtered_df = filtered_df.sort_values('amount', ascending=True)
        elif sort_by == "Category":
            filtered_df = filtered_df.sort_values('category')
    except:
        pass  # If sorting fails, continue without it

    # Bulk operations above the table
    st.subheader("🔧 Bulk Operations")

    # Initialize selection state in session state if not exists
    if 'selected_transactions' not in st.session_state:
        st.session_state.selected_transactions = []

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("✅ Select All Filtered", key="select_all_btn"):
            st.session_state.selected_transactions = list(range(len(filtered_df)))
            st.rerun()

    with col2:
        if st.button("❌ Clear All Selections", key="clear_all_btn"):
            st.session_state.selected_transactions = []
            st.rerun()

    with col3:
        selected_count = len(st.session_state.selected_transactions)
        st.write(f"Selected: {selected_count} transactions")

    with col4:
        if st.button("📤 Export Selected", key="export_btn"):
            if selected_count > 0:
                selected_data = filtered_df.iloc[st.session_state.selected_transactions]
                csv = selected_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"selected_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No transactions selected for export")

    # Bulk operations for selected transactions
    if selected_count > 0:
        st.write(f"**Operations for {selected_count} selected transactions:**")

        col1, col2 = st.columns(2)

        with col1:
            bulk_category = st.selectbox("Bulk Update Category", ai_cat.get_all_categories(), key="bulk_cat")
            if st.button("🔄 Update Selected Categories", key="bulk_update_btn"):
                selected_indices = filtered_df.iloc[st.session_state.selected_transactions].index
                changes_made = 0
                for idx in selected_indices:
                    if idx < len(st.session_state.transactions):
                        st.session_state.transactions.loc[idx, 'category'] = bulk_category
                        st.session_state.transactions.loc[idx, 'ai_categorized'] = False
                        changes_made += 1

                if changes_made > 0:
                    data_mgr.save_transactions(st.session_state.transactions)
                    st.success(f"✅ Updated {changes_made} transactions to {bulk_category}")
                    st.session_state.selected_transactions = []  # Clear selection
                    st.rerun()
                else:
                    st.info("No changes were made.")

        with col2:
            if st.button("🤖 Analyze Selected with AI", key="bulk_ai_btn"):
                selected_indices = filtered_df.iloc[st.session_state.selected_transactions].index
                analyzed_count = 0

                with st.spinner(f"Analyzing {len(selected_indices)} transactions..."):
                    progress_bar = st.progress(0)
                    analysis_results = []

                    for i, idx in enumerate(selected_indices):
                        if idx < len(st.session_state.transactions):
                            row = st.session_state.transactions.loc[idx]
                            analysis = ai_cat.analyze_narration_with_ai(row['narration'])

                            analysis_results.append({
                                'idx': idx,
                                'narration': row['narration'],
                                'analysis': analysis
                            })

                            progress_bar.progress((i + 1) / len(selected_indices))

                    # Display analysis results
                    if analysis_results:
                        st.subheader("AI Analysis Results")
                        for result in analysis_results:
                            analysis = result['analysis']
                            if analysis and analysis.get('confidence', 0) > 0:
                                with st.expander(f"📝 {result['narration'][:50]}..."):
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.write(f"**Purpose:** {analysis.get('purpose', 'N/A')}")
                                    with col2:
                                        st.write(f"**Suggested Category:** {analysis.get('suggested_category', 'Others')}")
                                    with col3:
                                        st.write(f"**Confidence:** {analysis.get('confidence', 0)}%")

                                    st.write(f"**Reasoning:** {analysis.get('reasoning', 'N/A')}")

                                    # Apply button for individual transaction
                                    if st.button(f"Apply Suggestion", key=f"apply_{result['idx']}"):
                                        st.session_state.transactions.loc[result['idx'], 'category'] = analysis.get('suggested_category', 'Others')
                                        st.session_state.transactions.loc[result['idx'], 'ai_categorized'] = True
                                        st.session_state.transactions.loc[result['idx'], 'similarity_score'] = analysis.get('confidence', 0)
                                        data_mgr.save_transactions(st.session_state.transactions)
                                        st.success(f"Applied suggestion for transaction")
                                        analyzed_count += 1
                                        st.rerun()
                            else:
                                st.warning(f"Could not analyze: {result['narration'][:50]}...")

                        # Bulk apply all suggestions
                        if st.button("Apply All AI Suggestions", key="bulk_apply_ai"):
                            for result in analysis_results:
                                analysis = result['analysis']
                                if analysis and analysis.get('confidence', 0) > 30:
                                    st.session_state.transactions.loc[result['idx'], 'category'] = analysis.get('suggested_category', 'Others')
                                    st.session_state.transactions.loc[result['idx'], 'ai_categorized'] = True
                                    st.session_state.transactions.loc[result['idx'], 'similarity_score'] = analysis.get('confidence', 0)
                                    analyzed_count += 1

                            if analyzed_count > 0:
                                data_mgr.save_transactions(st.session_state.transactions)
                                st.success(f"✅ Applied {analyzed_count} AI suggestions")
                                st.session_state.selected_transactions = []  # Clear selection
                                st.rerun()
                    else:
                        st.warning("No analysis results to display")

    # Transaction table display with selection
    st.subheader("📊 Transaction Table")
    
    # Add save button
    col_save1, col_save2, col_save3 = st.columns([1, 2, 1])
    with col_save2:
        if st.button("💾 Save Changes", type="primary", use_container_width=True):
            if 'transaction_updates' in st.session_state and st.session_state.transaction_updates:
                # Apply all pending updates
                for idx, updates in st.session_state.transaction_updates.items():
                    for field, value in updates.items():
                        st.session_state.transactions.loc[idx, field] = value
                
                # Save to file
                data_mgr.save_transactions(st.session_state.transactions)
                st.success(f"✅ Saved {len(st.session_state.transaction_updates)} transaction updates")
                
                # Clear pending updates
                st.session_state.transaction_updates = {}
                st.rerun()
            else:
                st.info("No pending changes to save")

    if not filtered_df.empty:
        # Prepare display dataframe
        display_df = filtered_df.copy()
        display_df = display_df.reset_index(drop=True)

        # Add selection column based on session state
        display_df['Select'] = display_df.index.isin(st.session_state.selected_transactions)

        # Add transaction type column - editable
        display_df['Type'] = display_df['amount'].apply(lambda x: 'Income' if x > 0 else 'Expense')

        # Improved AI Status with 3 categories
        display_df['AI Status'] = display_df.apply(lambda row: 
            f"AI ({row.get('similarity_score', 0):.0f}%)" if row.get('ai_categorized', False)
            else "Manual", axis=1)

        # Truncate narration for display
        display_df['Short Narration'] = display_df['narration'].apply(lambda x: x[:50] + "..." if len(x) > 50 else x)

        # Format amount with negative values in red and parentheses
        display_df['Amount (₹)'] = display_df['amount'].apply(lambda x: f"₹{x:,.2f}" if x >= 0 else f"(₹{abs(x):,.2f})")

        # Convert date to string to avoid compatibility issues
        display_df['Date'] = display_df['date'].astype(str)

        # Select columns for display
        display_columns = ['Select', 'Date', 'Short Narration', 'Amount (₹)', 'category', 'Type', 'AI Status']

        # Initialize edited state in session state if not exists
        if 'transaction_edits' not in st.session_state:
            st.session_state.transaction_edits = {}

        # Create editable dataframe without triggering automatic saves
        edited_df = st.data_editor(
            display_df[display_columns],
            column_config={
                "Select": st.column_config.CheckboxColumn("Select"),
                "Date": st.column_config.TextColumn("Date"),
                "Short Narration": st.column_config.TextColumn("Narration", width="large"),
                "Amount (₹)": st.column_config.TextColumn("Amount"),
                "category": st.column_config.SelectboxColumn("Category", options=ai_cat.get_all_categories()),
                "Type": st.column_config.SelectboxColumn("Type", options=["Income", "Expense", "Asset", "Liability"]),
                "AI Status": st.column_config.SelectboxColumn("AI Status", options=["Manual", "AI", "Software"])
            },
            use_container_width=True,
            hide_index=True,
            key="transaction_table",
            disabled=False
        )

        # Update selection state based on checkbox changes (without triggering rerun)
        if edited_df is not None:
            new_selection = [i for i, selected in enumerate(edited_df['Select']) if selected]
            # Store selection changes
            st.session_state.selected_transactions = new_selection
            
            # Store edits temporarily without applying them
            st.session_state.transaction_edits = edited_df.to_dict('records')

        # Save button for applying all changes
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save All Changes", key="save_changes_btn", type="primary"):
                changes_made = 0
                if 'transaction_edits' in st.session_state and st.session_state.transaction_edits:
                    for i, edited_row_dict in enumerate(st.session_state.transaction_edits):
                        if i < len(filtered_df):
                            original_idx = filtered_df.index[i]
                            original_category = filtered_df.loc[original_idx, 'category']
                            original_type = 'Income' if filtered_df.loc[original_idx, 'amount'] > 0 else 'Expense'

                            # Update category if changed
                            if edited_row_dict['category'] != original_category:
                                st.session_state.transactions.loc[original_idx, 'category'] = edited_row_dict['category']
                                audit_log.log_categorization_change(
                                    str(filtered_df.loc[original_idx, 'narration']), 
                                    original_category, 
                                    edited_row_dict['category']
                                )
                                changes_made += 1

                            # Update type if changed
                            if edited_row_dict['Type'] != original_type:
                                # Update transaction type based on selection
                                if edited_row_dict['Type'] == 'Income' and st.session_state.transactions.loc[original_idx, 'amount'] < 0:
                                    st.session_state.transactions.loc[original_idx, 'amount'] = abs(st.session_state.transactions.loc[original_idx, 'amount'])
                                elif edited_row_dict['Type'] == 'Expense' and st.session_state.transactions.loc[original_idx, 'amount'] > 0:
                                    st.session_state.transactions.loc[original_idx, 'amount'] = -abs(st.session_state.transactions.loc[original_idx, 'amount'])
                                changes_made += 1

                    if changes_made > 0:
                        data_mgr.save_transactions(st.session_state.transactions)
                        st.success(f"✅ Saved {changes_made} changes!")
                        # Clear the edits after saving
                        st.session_state.transaction_edits = {}
                        st.rerun()
                    else:
                        st.info("No changes detected to save.")
                else:
                    st.info("No changes to save.")

        with col2:
            if st.button("🔄 Reset Edits", key="reset_edits_btn"):
                st.session_state.transaction_edits = {}
                st.session_state.selected_transactions = []
                st.success("Edits reset!")
                st.rerun()
    else:
        st.info("No transactions match the current filters.")

def render_category_management_view(ai_cat, audit_log):
    """Render the category management view"""
    st.header("🏷️ Category Management")

    # Custom category creation
    st.subheader("Create Custom Category")
    col1, col2, col3 = st.columns(3)

    with col1:
        new_category_name = st.text_input("Category Name", placeholder="e.g., Gym Membership")

    with col2:
        new_category_keywords = st.text_input("Keywords (comma-separated)", placeholder="e.g., gym, fitness, workout")

    with col3:
        category_type = st.selectbox("Category Type", ["income", "expense", "asset", "liability"], index=1)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Add Custom Category"):
            if new_category_name and new_category_keywords:
                keywords_list = [kw.strip() for kw in new_category_keywords.split(',')]
                ai_cat.add_custom_category(new_category_name, keywords_list, category_type)
                audit_log.log_category_creation(new_category_name, keywords_list)
                st.success(f"Added custom category: {new_category_name} ({category_type})")
                st.rerun()
            else:
                st.error("Please provide both category name and keywords")

    with col2:
        if st.button("🔄 Refresh Transactions", help="Re-categorize all transactions with new categories"):
            if not st.session_state.transactions.empty:
                # Clear the categorization cache to force fresh categorization
                ai_cat.categorization_cache["patterns"] = {}
                ai_cat._save_categorization_cache()

                # Reload and update category patterns
                ai_cat.category_patterns = ai_cat._initialize_category_patterns()

                # Re-categorize all transactions
                updated_df = ai_cat.categorize_transactions(st.session_state.transactions)
                st.session_state.transactions = updated_df
                from utils.data_manager import DataManager
                data_mgr = DataManager()
                data_mgr.save_transactions(updated_df)
                st.success("✅ All transactions refreshed with updated categories!")
                st.rerun()
            else:
                st.warning("No transactions available to refresh.")

    # Display existing categories
    st.subheader("Existing Categories")

    # Default categories
    st.write("**Default Categories (Editable):**")
    default_categories = ai_cat.get_default_categories()

    # Ensure we have a dictionary to iterate over
    if isinstance(default_categories, dict):
        categories_to_process = default_categories
    else:
        # Convert list to dict format for processing
        categories_to_process = {}
        for category in default_categories:
            categories_to_process[category] = {
                "keywords": ai_cat.categories.get(category, []),
                "type": "expense"
            }

    for category, category_data in categories_to_process.items():
        keywords = category_data.get('keywords', [])
        category_type = category_data.get('type', 'expense')

        with st.expander(f"📝 {category} ({category_type}) - Default"):
            col1, col2 = st.columns(2)

            with col1:
                new_keywords = st.text_input(
                    "Keywords (comma-separated)", 
                    value=', '.join(keywords),
                    key=f"edit_default_keywords_{category}"
                )

            with col2:
                new_type = st.selectbox(
                    "Type", 
                    ["income", "expense", "asset", "liability"],
                    index=["income", "expense", "asset", "liability"].index(category_type),
                    key=f"edit_default_type_{category}"
                )

            if st.button("💾 Update Default", key=f"update_default_{category}"):
                if new_keywords.strip():
                    new_keywords_list = [kw.strip() for kw in new_keywords.split(',')]
                    ai_cat.update_default_category(category, new_keywords_list, new_type)
                    audit_log.log_category_creation(f"{category} (default updated)", new_keywords_list)
                    st.success(f"Updated default category: {category}")
                    st.rerun()
                else:
                    st.error("Keywords cannot be empty")

    # Custom categories
    st.write("**Custom Categories:**")
    custom_categories = ai_cat.get_custom_categories()

    if custom_categories:
        for category, category_data in custom_categories.items():
            # Handle both old and new format
            if isinstance(category_data, dict):
                keywords = category_data.get('keywords', [])
                category_type = category_data.get('type', 'expense')
            else:
                keywords = category_data if isinstance(category_data, list) else []
                category_type = 'expense'

            with st.expander(f"📝 {category} ({category_type})"):
                col1, col2 = st.columns(2)

                with col1:
                    new_keywords = st.text_input(
                        "Keywords (comma-separated)", 
                        value=', '.join(keywords),
                        key=f"edit_keywords_{category}"
                    )

                with col2:
                    new_type = st.selectbox(
                        "Type", 
                        ["income", "expense", "asset", "liability"],
                        index=["income", "expense", "asset", "liability"].index(category_type),
                        key=f"edit_type_{category}"
                    )

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Update", key=f"update_{category}"):
                        if new_keywords.strip():
                            new_keywords_list = [kw.strip() for kw in new_keywords.split(',')]
                            ai_cat.update_custom_category(category, new_keywords_list, new_type)
                            audit_log.log_category_creation(f"{category} (updated)", new_keywords_list)
                            st.success(f"Updated category: {category}")
                            st.rerun()
                        else:
                            st.error("Keywords cannot be empty")

                with col2:
                    if st.button("🗑️ Delete", key=f"del_{category}"):
                        ai_cat.delete_custom_category(category)
                        audit_log.log_category_deletion(category)
                        st.success(f"Deleted category: {category}")
                        st.rerun()
    else:
        st.info("No custom categories created yet.")

    # AI cache management is hidden from users for better UX

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

    # Check authentication
    if not st.session_state.authenticated:
        render_login()
        return

    # Initialize components with user-specific data
    (pdf_proc, excel_proc, trans_parser, ai_cat, data_mgr, chart_gen, 
     audit_log, rules_engine) = init_components(st.session_state.user_data_dir)

    # App title and description with logout
    col1, col2 = st.columns([4, 1])
    with col1:
        st.header("🏦 Bank Statement Analyzer")
        st.markdown("Upload your bank statements and get intelligent transaction categorization with AI learning capabilities")
    with col2:
        st.write(f"👤 {st.session_state.username}")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.user_data_dir = None
            st.session_state.transactions = pd.DataFrame()
            st.rerun()

    # Header with navigation buttons
    st.markdown("""
    <style>
    .nav-header {
        padding: 1rem 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 2rem;
    }
    .nav-button {
        margin-right: 10px;
        margin-bottom: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="nav-header">', unsafe_allow_html=True)
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

        with col1:
            if st.button("📄 Upload & Process", key="nav_upload", help="Upload and process bank statements"):
                st.session_state.current_view = "Upload & Process"
        with col2:
            if st.button("📊 Dashboard", key="nav_dashboard", help="View financial dashboard"):
                st.session_state.current_view = "Dashboard"
        with col3:
            if st.button("📋 Transactions", key="nav_transactions", help="View and edit transaction details"):
                st.session_state.current_view = "Transaction Details"
        with col4:
            if st.button("🏷️ Categories", key="nav_categories", help="Manage categories"):
                st.session_state.current_view = "Category Management"
        with col5:
            if st.button("📜 Audit Trail", key="nav_audit", help="View audit logs"):
                st.session_state.current_view = "Audit Trail"
        with col6:
            if st.button("📊 Balance Sheet", key="nav_balance", help="View balance sheet"):
                st.session_state.current_view = "Balance Sheet"
        with col7:
            if st.button("⚙️ Settings", key="nav_settings", help="Application settings"):
                st.session_state.current_view = "Settings"

        st.markdown('</div>', unsafe_allow_html=True)

    selected_view = st.session_state.current_view

    # Route to appropriate view
    if selected_view == "Upload & Process":
        render_upload_view(pdf_proc, excel_proc, trans_parser, ai_cat, data_mgr, audit_log)
    elif selected_view == "Dashboard":
        render_dashboard_view(data_mgr, chart_gen, ai_cat, audit_log)
    elif selected_view == "Transaction Details":
        render_transaction_details_view(data_mgr, ai_cat, audit_log)
    elif selected_view == "Category Management":
        render_category_management_view(ai_cat, audit_log)
    elif selected_view == "Audit Trail":
        render_audit_trail_view(audit_log)
    elif selected_view == "Balance Sheet":
        render_balance_sheet_view(data_mgr, ai_cat)
    elif selected_view == "Settings":
        render_settings_view(data_mgr, ai_cat, audit_log, rules_engine)

    # Footer
    st.markdown("---")
    st.markdown("💡 **Tip:** The AI learns from your corrections and gets better over time. All data is stored locally on your device.")

def render_balance_sheet_view(data_mgr, ai_cat):
    """Render balance sheet view for assets and liabilities"""
    st.header("📊 Balance Sheet")

    if st.session_state.transactions.empty:
        existing_transactions = data_mgr.load_transactions()
        if not existing_transactions.empty:
            st.session_state.transactions = existing_transactions

    df = st.session_state.transactions

    if df.empty:
        st.info("No transactions available. Please upload and process bank statements first.")
        return

    # Separate assets and liabilities
    all_categories = ai_cat.get_all_categories()
    asset_categories = [cat for cat in all_categories if ai_cat.get_category_type(cat) == 'asset']
    liability_categories = [cat for cat in all_categories if ai_cat.get_category_type(cat) == 'liability']

    assets_df = df[df['category'].isin(asset_categories)]
    liabilities_df = df[df['category'].isin(liability_categories)]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💰 Assets")
        if not assets_df.empty:
            asset_summary = assets_df.groupby('category')['amount'].sum().reset_index()
            asset_summary['amount'] = asset_summary['amount'].abs()
            asset_summary = asset_summary.sort_values('amount', ascending=False)

            total_assets = asset_summary['amount'].sum()
            st.metric("Total Assets", f"₹{total_assets:,.2f}")

            for _, row in asset_summary.iterrows():
                with st.expander(f"{row['category']} - ₹{row['amount']:,.2f}"):
                    category_transactions = assets_df[assets_df['category'] == row['category']]
                    st.dataframe(category_transactions[['date', 'narration', 'amount']], use_container_width=True)
        else:
            st.info("No asset transactions found")

    with col2:
        st.subheader("📉 Liabilities")
        if not liabilities_df.empty:
            liability_summary = liabilities_df.groupby('category')['amount'].sum().reset_index()
            liability_summary['amount'] = liability_summary['amount'].abs()
            liability_summary = liability_summary.sort_values('amount', ascending=False)

            total_liabilities = liability_summary['amount'].sum()
            st.metric("Total Liabilities", f"₹{total_liabilities:,.2f}")

            for _, row in liability_summary.iterrows():
                with st.expander(f"{row['category']} - ₹{row['amount']:,.2f}"):
                    category_transactions = liabilities_df[liabilities_df['category'] == row['category']]
                    st.dataframe(category_transactions[['date', 'narration', 'amount']], use_container_width=True)
        else:
            st.info("No liability transactions found")

    # Net Worth calculation
    if not assets_df.empty or not liabilities_df.empty:
        total_assets = assets_df['amount'].abs().sum() if not assets_df.empty else 0
        total_liabilities = liabilities_df['amount'].abs().sum() if not liabilities_df.empty else 0
        net_worth = total_assets - total_liabilities

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Assets", f"₹{total_assets:,.2f}")
        with col2:
            st.metric("Total Liabilities", f"₹{total_liabilities:,.2f}")
        with col3:
            st.metric("Net Worth", f"₹{net_worth:,.2f}", delta=None)

# Run the app
if __name__ == "__main__":
    main()