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
    excel_processor = ExcelProcessor()
    transaction_parser = TransactionParser()
    ai_categorizer = AICategorizer(user_data_dir)
    data_manager = DataManager(user_data_dir)
    chart_generator = ChartGenerator()
    audit_logger = AuditLogger()
    rules_engine = RulesEngine()
    return (excel_processor, transaction_parser, ai_categorizer, data_manager, 
            chart_generator, audit_logger, rules_engine)

def _rematch_suspense_transactions(ai_cat, data_mgr):
    """Helper function to rematch suspense transactions when new categories are added - uses only cached data"""
    try:
        # Get current transactions
        df = st.session_state.transactions
        
        # Find suspense transactions
        suspense_mask = df['category'] == 'Suspense'
        suspense_transactions = df[suspense_mask].copy()
        
        if not suspense_transactions.empty:
            print(f"Rematching {len(suspense_transactions)} suspense transactions using cached data only")
            
            # Use only cached patterns and rule-based matching (no AI calls)
            for idx, row in suspense_transactions.iterrows():
                narration = str(row['narration']).lower().strip()
                
                # Check cache first
                if narration in ai_cat.categorization_cache.get("patterns", {}):
                    new_category = ai_cat.categorization_cache["patterns"][narration]
                    st.session_state.transactions.loc[idx, 'category'] = new_category
                    st.session_state.transactions.loc[idx, 'processing_type'] = 'Software'
                    st.session_state.transactions.loc[idx, 'confidence'] = 95
                    print(f"Cache rematch: '{row['narration']}' -> '{new_category}'")
                    continue
                
                # Check user corrections cache
                if narration in ai_cat.categorization_cache.get("corrections", {}):
                    new_category = ai_cat.categorization_cache["corrections"][narration]
                    st.session_state.transactions.loc[idx, 'category'] = new_category
                    st.session_state.transactions.loc[idx, 'processing_type'] = 'Software'
                    st.session_state.transactions.loc[idx, 'confidence'] = 100
                    print(f"Correction cache rematch: '{row['narration']}' -> '{new_category}'")
                    continue
                
                # Use rule-based matching only
                cleaned_narration = ai_cat._clean_narration_for_matching(row['narration'])
                category, similarity = ai_cat._apply_rule_based_matching(cleaned_narration)
                if similarity > 50 and category != "Others":
                    st.session_state.transactions.loc[idx, 'category'] = category
                    st.session_state.transactions.loc[idx, 'processing_type'] = 'Software'
                    st.session_state.transactions.loc[idx, 'confidence'] = similarity
                    print(f"Rule-based rematch: '{row['narration']}' -> '{category}' ({similarity}%)")
            
            # Save the updated transactions
            data_mgr.save_transactions(st.session_state.transactions)
            
    except Exception as e:
        print(f"Error rematching suspense transactions: {str(e)}")

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
    if 'pending_updates' not in st.session_state:
        st.session_state.pending_updates = {}

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

def render_upload_view(excel_proc, trans_parser, ai_cat, data_mgr, audit_log):
    """Render the upload and process view"""
    st.subheader("📄 Upload Bank Statements")

    # Only Excel/CSV upload
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
                    # Process Excel/CSV files
                    transactions = excel_proc.extract_transactions(file, file.name)

                    if not transactions.empty:
                        # AI categorization with new logic
                        categorized_transactions = ai_cat.categorize_transactions_new_logic(transactions)
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
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Transactions", len(combined_df))
                with col2:
                    income_total = combined_df[combined_df['amount'] > 0]['amount'].sum()
                    st.metric("Total Income", f"₹{income_total:,.2f}")
                with col3:
                    expense_total = abs(combined_df[combined_df['amount'] < 0]['amount'].sum())
                    st.metric("Total Expenses", f"₹{expense_total:,.2f}")
                with col4:
                    assets_total = combined_df[combined_df['category'] == 'Assets']['amount'].abs().sum()
                    st.metric("Total Assets", f"₹{assets_total:,.2f}")

def render_dashboard_view(data_mgr, ai_cat, audit_log):
    """Render the dashboard view with new requirements"""
    st.subheader("📊 Financial Dashboard")

    # Load existing transactions if available
    if st.session_state.transactions.empty:
        existing_transactions = data_mgr.load_transactions()
        if not existing_transactions.empty:
            st.session_state.transactions = existing_transactions

    df = st.session_state.transactions

    if df.empty:
        st.info("No transactions available. Please upload and process bank statements first.")
        return

    # Financial Totals - No Charts
    col1, col2, col3, col4 = st.columns(4)

    income_df = df[df['amount'] > 0]
    expense_df = df[df['amount'] < 0]
    assets_df = df[df['category'] == 'Assets']
    liabilities_df = df[df['category'] == 'Liabilities']

    total_income = income_df['amount'].sum()
    total_expenses = abs(expense_df['amount'].sum())
    total_assets = assets_df['amount'].abs().sum()
    total_liabilities = liabilities_df['amount'].abs().sum()

    with col1:
        st.metric("Total Income", f"₹{total_income:,.2f}")
    with col2:
        st.metric("Total Expenses", f"₹{total_expenses:,.2f}")
    with col3:
        st.metric("Total Assets", f"₹{total_assets:,.2f}")
    with col4:
        st.metric("Total Liabilities", f"₹{total_liabilities:,.2f}")

    # Suspense Table with AI Search
    st.subheader("⚠️ Suspense Transactions")
    suspense_df = df[df['category'] == 'Suspense'].copy().reset_index(drop=True)

    if not suspense_df.empty:
        # AI search for suspense
        col1, col2 = st.columns(2)
        with col1:
            suspense_search = st.text_input("🔍 Search suspense transactions", placeholder="Type to search...", key="suspense_search")
        with col2:
            if st.button("🤖 AI Search Selected Suspense"):
                # Only search selected suspense transactions
                selected_suspense = []
                if 'suspense_editor' in st.session_state and hasattr(st.session_state.suspense_editor, 'edited_rows'):
                    for i, row in enumerate(st.session_state.suspense_editor.edited_rows):
                        if row.get('Select', False) and i < len(filtered_suspense):
                            selected_suspense.append(filtered_suspense.iloc[i])
                
                if selected_suspense:
                    st.session_state.ai_search_suspense = True
                    st.session_state.selected_suspense_for_ai = selected_suspense
                else:
                    st.warning("Please select transactions first using checkboxes.")
                
        # Apply search filter
        filtered_suspense = suspense_df.copy()
        if suspense_search:
            filtered_suspense = filtered_suspense[filtered_suspense['narration'].str.contains(suspense_search, case=False, na=False)]

        # Format amounts with red brackets for negative
        filtered_suspense['Amount Display'] = filtered_suspense['amount'].apply(
            lambda x: f"₹{x:,.2f}" if x >= 0 else f"(₹{abs(x):,.2f})"
        )

        # Add selection column
        filtered_suspense['Select'] = False

        # Add confidence column and format as percentage
        display_suspense = filtered_suspense.copy()
        if 'confidence' in display_suspense.columns:
            display_suspense['Confidence'] = display_suspense['confidence'].apply(lambda x: f"{x:.0f}%" if pd.notnull(x) else "0%")
        else:
            display_suspense['Confidence'] = "0%"

        # Editable table
        edited_suspense = st.data_editor(
            display_suspense[['Select', 'date', 'narration', 'Amount Display', 'category', 'processing_type', 'Confidence']],
            column_config={
                "Select": st.column_config.CheckboxColumn("Select"),
                "date": st.column_config.TextColumn("Date"),
                "narration": st.column_config.TextColumn("Narration", width="large"),
                "Amount Display": st.column_config.TextColumn("Amount"),
                "category": st.column_config.SelectboxColumn("Category", options=ai_cat.get_all_categories()),
                "processing_type": st.column_config.TextColumn("Processed By", disabled=True),
                "Confidence": st.column_config.TextColumn("Confidence", disabled=True)
            },
            use_container_width=True,
            hide_index=True,
            key="suspense_editor"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("💾 Save Suspense Changes"):
                # Use iloc to avoid index issues
                changes_made = 0
                for i in range(len(edited_suspense)):
                    if i < len(filtered_suspense):
                        # Find original transaction in main dataframe
                        original_transaction = filtered_suspense.iloc[i]
                        mask = (st.session_state.transactions['date'] == original_transaction['date']) & \
                               (st.session_state.transactions['narration'] == original_transaction['narration']) & \
                               (st.session_state.transactions['amount'] == original_transaction['amount'])
                        
                        if edited_suspense.iloc[i]['category'] != original_transaction['category']:
                            # Update main dataframe
                            st.session_state.transactions.loc[mask, 'category'] = edited_suspense.iloc[i]['category']
                            # Auto-learn from user correction
                            ai_cat.learn_from_correction(
                                original_transaction['narration'], 
                                edited_suspense.iloc[i]['category']
                            )
                            changes_made += 1

                if changes_made > 0:
                    data_mgr.save_transactions(st.session_state.transactions)
                    st.success(f"Saved {changes_made} changes and AI learned from corrections!")
                    # Trigger rematch of suspense transactions
                    _rematch_suspense_transactions(ai_cat, data_mgr)
                    st.rerun()

        with col2:
            if st.button("✅ Mark Selected as Accepted"):
                selected_count = sum(1 for i in range(len(edited_suspense)) if edited_suspense.iloc[i]['Select'])
                if selected_count > 0:
                    # Remove selected transactions from suspense without triggering AI
                    for i in range(len(edited_suspense)):
                        if i < len(filtered_suspense) and edited_suspense.iloc[i]['Select']:
                            original_transaction = filtered_suspense.iloc[i]
                            mask = (st.session_state.transactions['date'] == original_transaction['date']) & \
                                   (st.session_state.transactions['narration'] == original_transaction['narration']) & \
                                   (st.session_state.transactions['amount'] == original_transaction['amount'])
                            
                            # Update category to remove from suspense and set processing type to Manual
                            matching_rows = st.session_state.transactions.loc[mask]
                            if not matching_rows.empty and matching_rows.iloc[0]['category'] == 'Suspense':
                                st.session_state.transactions.loc[mask, 'category'] = 'Others'
                                st.session_state.transactions.loc[mask, 'processing_type'] = 'Manual'
                                st.session_state.transactions.loc[mask, 'confidence'] = 0

                    data_mgr.save_transactions(st.session_state.transactions)
                    st.success(f"Marked {selected_count} transactions as accepted!")
                    st.rerun()

        with col3:
            if st.button("🔄 Rematch All Suspense"):
                _rematch_suspense_transactions(ai_cat, data_mgr)
                st.success("Rematched all suspense transactions!")
                st.rerun()

        # Show AI search results for selected suspense transactions only
        if st.session_state.get('ai_search_suspense', False) and st.session_state.get('selected_suspense_for_ai', []):
            st.subheader("🤖 AI Analysis for Selected Suspense Transactions")
            ai_results = []
            
            # Only process selected transactions
            for row in st.session_state.selected_suspense_for_ai:
                # Check if already analyzed to avoid re-analysis
                narration_lower = str(row['narration']).lower().strip()
                if narration_lower not in ai_cat.categorization_cache.get("patterns", {}):
                    analysis = ai_cat.analyze_narration_with_ai(row['narration'])
                    ai_results.append({
                        'narration': row['narration'],
                        'current_category': row['category'],
                        'ai_suggested_category': analysis.get('suggested_category', 'Others'),
                        'confidence': analysis.get('confidence', 0),
                        'reasoning': analysis.get('reasoning', 'No reasoning provided'),
                        'select': False
                    })
                else:
                    # Use cached result
                    cached_category = ai_cat.categorization_cache["patterns"][narration_lower]
                    ai_results.append({
                        'narration': row['narration'],
                        'current_category': row['category'],
                        'ai_suggested_category': cached_category,
                        'confidence': 95,
                        'reasoning': 'Cached result from previous analysis',
                        'select': False
                    })

            if ai_results:
                ai_df = pd.DataFrame(ai_results)
                # Format confidence as percentage
                ai_df['confidence_display'] = ai_df['confidence'].apply(lambda x: f"{x:.0f}%")
                
                edited_ai_suspense = st.data_editor(
                    ai_df[['select', 'narration', 'current_category', 'ai_suggested_category', 'confidence_display', 'reasoning']],
                    column_config={
                        "select": st.column_config.CheckboxColumn("Select"),
                        "narration": st.column_config.TextColumn("Narration", width="large"),
                        "current_category": st.column_config.TextColumn("Current Category"),
                        "ai_suggested_category": st.column_config.TextColumn("AI Suggested"),
                        "confidence_display": st.column_config.TextColumn("Confidence", disabled=True),
                        "reasoning": st.column_config.TextColumn("AI Reasoning", width="large")
                    },
                    use_container_width=True,
                    hide_index=True,
                    key="ai_suspense_editor"
                )

                if st.button("✅ Accept Selected AI Suspense Suggestions"):
                    selected_count = 0
                    for i, row in edited_ai_suspense.iterrows():
                        if row['select']:
                            original_row = ai_df.iloc[i]
                            # Find and update in main dataframe
                            mask = (st.session_state.transactions['narration'] == original_row['narration']) & \
                                   (st.session_state.transactions['category'] == 'Suspense')
                            st.session_state.transactions.loc[mask, 'category'] = original_row['ai_suggested_category']
                            st.session_state.transactions.loc[mask, 'confidence'] = original_row['confidence']
                            st.session_state.transactions.loc[mask, 'processing_type'] = 'AI'
                            
                            # Cache the result to prevent re-analysis
                            narration_lower = str(original_row['narration']).lower().strip()
                            ai_cat.categorization_cache["patterns"][narration_lower] = original_row['ai_suggested_category']
                            selected_count += 1

                    if selected_count > 0:
                        ai_cat._save_categorization_cache()
                        data_mgr.save_transactions(st.session_state.transactions)
                        st.success(f"Applied {selected_count} AI suggestions!")
                        st.session_state.ai_search_suspense = False
                        del st.session_state.selected_suspense_for_ai
                        st.rerun()
    else:
        st.info("No transactions in suspense.")

    # Suspected Entries (Top transactions covering 80% of total)
    st.subheader("🔍 Suspected High-Value Transactions")

    # Calculate cumulative percentage
    df_sorted = df.copy()
    df_sorted['abs_amount'] = df_sorted['amount'].abs()
    df_sorted = df_sorted.sort_values('abs_amount', ascending=False)
    df_sorted['cumulative_pct'] = df_sorted['abs_amount'].cumsum() / df_sorted['abs_amount'].sum() * 100

    # Get top transactions covering 80%
    suspected_df = df_sorted[df_sorted['cumulative_pct'] <= 80].copy().reset_index(drop=True)

    if not suspected_df.empty:
        suspected_df['Amount Display'] = suspected_df['amount'].apply(
            lambda x: f"₹{x:,.2f}" if x >= 0 else f"(₹{abs(x):,.2f})"
        )
        suspected_df['Select'] = False

        edited_suspected = st.data_editor(
            suspected_df[['Select', 'date', 'narration', 'Amount Display', 'category', 'processing_type']],
            column_config={
                "Select": st.column_config.CheckboxColumn("Select"),
                "date": st.column_config.TextColumn("Date"),
                "narration": st.column_config.TextColumn("Narration", width="large"),
                "Amount Display": st.column_config.TextColumn("Amount"),
                "category": st.column_config.SelectboxColumn("Category", options=ai_cat.get_all_categories()),
                "processing_type": st.column_config.TextColumn("Processed By", disabled=True)
            },
            use_container_width=True,
            hide_index=True,
            key="suspected_editor"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("💾 Save Suspected Changes"):
                changes_made = 0
                for i in range(len(edited_suspected)):
                    if i < len(suspected_df):
                        original_transaction = suspected_df.iloc[i]
                        mask = (st.session_state.transactions['date'] == original_transaction['date']) & \
                               (st.session_state.transactions['narration'] == original_transaction['narration']) & \
                               (st.session_state.transactions['amount'] == original_transaction['amount'])
                        
                        if edited_suspected.iloc[i]['category'] != original_transaction['category']:
                            st.session_state.transactions.loc[mask, 'category'] = edited_suspected.iloc[i]['category']
                            ai_cat.learn_from_correction(
                                original_transaction['narration'], 
                                edited_suspected.iloc[i]['category']
                            )
                            changes_made += 1

                if changes_made > 0:
                    data_mgr.save_transactions(st.session_state.transactions)
                    st.success(f"Saved {changes_made} suspected transaction changes!")
                    st.rerun()

        with col2:
            if st.button("✅ Accept Selected Suspected"):
                selected_count = 0
                for i in range(len(edited_suspected)):
                    if i < len(suspected_df) and edited_suspected.iloc[i]['Select']:
                        selected_count += 1

                if selected_count > 0:
                    st.success(f"Accepted {selected_count} suspected transactions!")

def render_transaction_details_view(data_mgr, ai_cat, audit_log):
    """Render the transaction details view with enhanced features"""
    st.subheader("📋 Transaction Details")

    if st.session_state.transactions.empty:
        existing_transactions = data_mgr.load_transactions()
        if not existing_transactions.empty:
            st.session_state.transactions = existing_transactions

    df = st.session_state.transactions

    # Ensure required columns exist
    if not df.empty:
        if 'processing_type' not in df.columns:
            df['processing_type'] = 'Manual'
        if 'confidence' not in df.columns:
            df['confidence'] = 0.0
        st.session_state.transactions = df

    if df.empty:
        st.info("No transactions available. Please upload and process bank statements first.")
        return

    # Convert date column to datetime for filtering
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')

    # Search and filters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        search_term = st.text_input("🔍 Search transactions", placeholder="Type to search...")

    with col2:
        categories = ['All'] + sorted(df['category'].unique().tolist())
        selected_category = st.selectbox("Filter by Category", categories)

    with col3:
        # Date range filter
        if not df_copy['date'].isna().all():
            min_date = df_copy['date'].min().date()
            max_date = df_copy['date'].max().date()
            
            date_from = st.date_input("From Date", value=min_date, min_value=min_date, max_value=max_date)
            date_to = st.date_input("To Date", value=max_date, min_value=min_date, max_value=max_date)
        else:
            date_from = None
            date_to = None

    with col4:
        selected_count = 0
        if st.button("🤖 AI Search Selected"):
            # Get selected transactions for AI search
            if 'selected_for_ai_search' in st.session_state:
                selected_count = len(st.session_state.selected_for_ai_search)
                if selected_count > 0:
                    st.session_state.ai_search_selected = True
                else:
                    st.warning("Please select transactions first using checkboxes below.")
            else:
                st.warning("Please select transactions first using checkboxes below.")

    # Apply filters
    filtered_df = df_copy.copy()
    
    if search_term:
        filtered_df = filtered_df[filtered_df['narration'].str.contains(search_term, case=False, na=False)]
    
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['category'] == selected_category]
    
    # Apply date filter
    if date_from and date_to and not filtered_df['date'].isna().all():
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= date_from) & 
            (filtered_df['date'].dt.date <= date_to)
        ]

    # Convert date back to string for display
    filtered_df['date'] = filtered_df['date'].dt.strftime('%Y-%m-%d')

    # Prepare display with negative amounts in red brackets
    if not filtered_df.empty:
        display_df = filtered_df.copy().reset_index(drop=True)
        display_df['Select'] = False
        display_df['Amount Display'] = display_df['amount'].apply(
            lambda x: f"₹{x:,.2f}" if x >= 0 else f"(₹{abs(x):,.2f})"
        )

        # Format confidence as percentage
        display_df['Confidence'] = display_df['confidence'].apply(lambda x: f"{x:.0f}%" if pd.notnull(x) else "0%")
        
        # Editable table
        edited_df = st.data_editor(
            display_df[['Select', 'date', 'narration', 'Amount Display', 'category', 'processing_type', 'Confidence']],
            column_config={
                "Select": st.column_config.CheckboxColumn("Select"),
                "date": st.column_config.TextColumn("Date"),
                "narration": st.column_config.TextColumn("Narration", width="large"),
                "Amount Display": st.column_config.TextColumn("Amount"),
                "category": st.column_config.SelectboxColumn("Category", options=ai_cat.get_all_categories()),
                "processing_type": st.column_config.TextColumn("Processed By"),
                "Confidence": st.column_config.TextColumn("Confidence", disabled=True)
            },
            use_container_width=True,
            hide_index=True,
            key="transaction_editor"
        )

        # Store selected transactions for AI search
        selected_transactions = []
        for i, row in edited_df.iterrows():
            if row['Select']:
                selected_transactions.append(display_df.iloc[i])
        st.session_state.selected_for_ai_search = selected_transactions

        # Control buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("💾 Save Changes", type="primary"):
                changes_made = 0
                for i in range(len(edited_df)):
                    if i < len(display_df):
                        original_transaction = display_df.iloc[i]
                        mask = (st.session_state.transactions['date'] == original_transaction['date']) & \
                               (st.session_state.transactions['narration'] == original_transaction['narration']) & \
                               (st.session_state.transactions['amount'] == original_transaction['amount'])
                        
                        if edited_df.iloc[i]['category'] != original_transaction['category']:
                            st.session_state.transactions.loc[mask, 'category'] = edited_df.iloc[i]['category']
                            ai_cat.learn_from_correction(
                                original_transaction['narration'], 
                                edited_df.iloc[i]['category']
                            )
                            changes_made += 1

                if changes_made > 0:
                    data_mgr.save_transactions(st.session_state.transactions)
                    st.success(f"Saved {changes_made} changes and AI learned from corrections!")
                    # Trigger rematch
                    _rematch_suspense_transactions(ai_cat, data_mgr)
                    st.rerun()

        with col2:
            selected_count = sum(1 for i in range(len(edited_df)) if edited_df.iloc[i]['Select'])
            if selected_count > 0:
                bulk_category = st.selectbox("Bulk update category", ai_cat.get_all_categories(), key="bulk_category")
                if st.button(f"🔄 Bulk Update ({selected_count})"):
                    for i in range(len(edited_df)):
                        if i < len(display_df) and edited_df.iloc[i]['Select']:
                            original_transaction = display_df.iloc[i]
                            mask = (st.session_state.transactions['date'] == original_transaction['date']) & \
                                   (st.session_state.transactions['narration'] == original_transaction['narration']) & \
                                   (st.session_state.transactions['amount'] == original_transaction['amount'])
                            
                            st.session_state.transactions.loc[mask, 'category'] = bulk_category

                    data_mgr.save_transactions(st.session_state.transactions)
                    st.success(f"Updated {selected_count} transactions!")
                    st.rerun()

        # Show AI search results for selected transactions
        if st.session_state.get('ai_search_selected', False) and st.session_state.get('selected_for_ai_search', []):
            st.subheader("🤖 AI Analysis for Selected Transactions")
            
            ai_results = []
            for transaction in st.session_state.selected_for_ai_search:
                # Check if already analyzed to avoid re-analysis
                narration_lower = str(transaction['narration']).lower().strip()
                if narration_lower not in ai_cat.categorization_cache.get("patterns", {}):
                    analysis = ai_cat.analyze_narration_with_ai(transaction['narration'])
                    ai_results.append({
                        'narration': transaction['narration'],
                        'current_category': transaction['category'],
                        'ai_suggested_category': analysis.get('suggested_category', 'Others'),
                        'confidence': analysis.get('confidence', 0),
                        'reasoning': analysis.get('reasoning', 'No reasoning provided'),
                        'select': False
                    })
                else:
                    # Use cached result
                    cached_category = ai_cat.categorization_cache["patterns"][narration_lower]
                    ai_results.append({
                        'narration': transaction['narration'],
                        'current_category': transaction['category'],
                        'ai_suggested_category': cached_category,
                        'confidence': 95,
                        'reasoning': 'Cached result from previous analysis',
                        'select': False
                    })

            if ai_results:
                ai_df = pd.DataFrame(ai_results)
                # Format confidence as percentage
                ai_df['confidence_display'] = ai_df['confidence'].apply(lambda x: f"{x:.0f}%")
                
                edited_ai_selected = st.data_editor(
                    ai_df[['select', 'narration', 'current_category', 'ai_suggested_category', 'confidence_display', 'reasoning']],
                    column_config={
                        "select": st.column_config.CheckboxColumn("Select"),
                        "narration": st.column_config.TextColumn("Narration", width="large"),
                        "current_category": st.column_config.TextColumn("Current Category"),
                        "ai_suggested_category": st.column_config.TextColumn("AI Suggested"),
                        "confidence_display": st.column_config.TextColumn("Confidence", disabled=True),
                        "reasoning": st.column_config.TextColumn("AI Reasoning", width="large")
                    },
                    use_container_width=True,
                    hide_index=True,
                    key="ai_selected_editor"
                )

                if st.button("✅ Accept Selected AI Suggestions"):
                    selected_ai_count = 0
                    for i, row in edited_ai_selected.iterrows():
                        if row['select']:
                            original_row = ai_df.iloc[i]
                            # Find and update in main dataframe
                            mask = (st.session_state.transactions['narration'] == original_row['narration'])
                            st.session_state.transactions.loc[mask, 'category'] = original_row['ai_suggested_category']
                            st.session_state.transactions.loc[mask, 'confidence'] = original_row['confidence']
                            st.session_state.transactions.loc[mask, 'processing_type'] = 'AI'
                            
                            # Cache the result to prevent re-analysis
                            narration_lower = str(original_row['narration']).lower().strip()
                            ai_cat.categorization_cache["patterns"][narration_lower] = original_row['ai_suggested_category']
                            selected_ai_count += 1

                    if selected_ai_count > 0:
                        ai_cat._save_categorization_cache()
                        data_mgr.save_transactions(st.session_state.transactions)
                        st.success(f"Applied {selected_ai_count} AI suggestions!")
                        st.session_state.ai_search_selected = False
                        del st.session_state.selected_for_ai_search
                        st.rerun()
    else:
        st.info("No transactions match the current filters.")

def render_ai_search_view(ai_cat, data_mgr):
    """Render AI search results view"""
    st.subheader("🤖 AI Transaction Analysis")

    df = st.session_state.transactions
    if df.empty:
        st.info("No transactions available.")
        return

    # Get uncategorized or low confidence transactions
    ai_candidates = df[
        (df['category'].isin(['Suspense', 'Others'])) | 
        (df['confidence'] < 50)
    ].copy()

    if ai_candidates.empty:
        st.info("No transactions need AI analysis.")
        return

    # AI analysis results
    st.write("AI suggestions for uncategorized transactions:")

    ai_results = []
    for _, row in ai_candidates.iterrows():
        analysis = ai_cat.analyze_narration_with_ai(row['narration'])
        ai_results.append({
            'narration': row['narration'],
            'current_category': row['category'],
            'ai_suggested_category': analysis.get('suggested_category', 'Others'),
            'confidence': analysis.get('confidence', 0),
            'reasoning': analysis.get('reasoning', 'No reasoning provided'),
            'select': False
        })

    if ai_results:
        ai_df = pd.DataFrame(ai_results)

        # Format confidence as percentage
        ai_df['confidence_display'] = ai_df['confidence'].apply(lambda x: f"{x:.0f}%")
        
        edited_ai = st.data_editor(
            ai_df[['select', 'narration', 'current_category', 'ai_suggested_category', 'confidence_display', 'reasoning']],
            column_config={
                "select": st.column_config.CheckboxColumn("Select"),
                "narration": st.column_config.TextColumn("Narration", width="large"),
                "current_category": st.column_config.TextColumn("Current Category"),
                "ai_suggested_category": st.column_config.TextColumn("AI Suggested"),
                "confidence_display": st.column_config.TextColumn("Confidence", disabled=True),
                "reasoning": st.column_config.TextColumn("AI Reasoning", width="large")
            },
            use_container_width=True,
            hide_index=True,
            key="ai_search_editor"
        )

        if st.button("✅ Accept Selected AI Suggestions"):
            selected_count = 0
            for i, row in edited_ai.iterrows():
                if row['select']:
                    # Find and update in main dataframe
                    mask = (st.session_state.transactions['narration'] == row['narration'])
                    st.session_state.transactions.loc[mask, 'category'] = row['ai_suggested_category']
                    st.session_state.transactions.loc[mask, 'confidence'] = row['confidence']
                    st.session_state.transactions.loc[mask, 'processing_type'] = 'AI'
                    selected_count += 1

            if selected_count > 0:
                data_mgr.save_transactions(st.session_state.transactions)
                st.success(f"Applied {selected_count} AI suggestions!")
                st.rerun()

def render_category_management_view(ai_cat, audit_log):
    """Render the category management view"""
    st.subheader("🏷️ Category Management")

    # Custom category creation
    st.write("**Create Custom Category**")
    col1, col2, col3 = st.columns(3)

    with col1:
        new_category_name = st.text_input("Category Name")

    with col2:
        new_category_keywords = st.text_input("Keywords (comma-separated)")

    with col3:
        category_type = st.selectbox("Category Type", ["income", "expense", "asset", "liability"], index=1)

    if st.button("Add Custom Category"):
        if new_category_name and new_category_keywords:
            keywords_list = [kw.strip() for kw in new_category_keywords.split(',')]
            ai_cat.add_custom_category(new_category_name, keywords_list, category_type)
            audit_log.log_category_creation(new_category_name, keywords_list)
            
            # Rematch suspense transactions with new category
            data_mgr_instance = DataManager(st.session_state.user_data_dir)
            _rematch_suspense_transactions(ai_cat, data_mgr_instance)
            
            st.success(f"Added custom category: {new_category_name} and rematched suspense transactions!")
            st.rerun()

    # Display existing categories
    st.write("**Existing Categories**")
    default_categories = ai_cat.get_default_categories()
    custom_categories = ai_cat.get_custom_categories()

    for category, details in {**default_categories, **custom_categories}.items():
        with st.expander(f"📝 {category}"):
            keywords = details.get('keywords', [])
            cat_type = details.get('type', 'expense')

            new_keywords = st.text_input(
                "Keywords", 
                value=', '.join(keywords),
                key=f"keywords_{category}"
            )

            new_type = st.selectbox(
                "Type", 
                ["income", "expense", "asset", "liability"],
                index=["income", "expense", "asset", "liability"].index(cat_type),
                key=f"type_{category}"
            )

            if st.button("Update", key=f"update_{category}"):
                keywords_list = [kw.strip() for kw in new_keywords.split(',')]
                ai_cat.update_category(category, keywords_list, new_type)
                
                # Rematch suspense transactions with updated category
                data_mgr_instance = DataManager(st.session_state.user_data_dir)
                _rematch_suspense_transactions(ai_cat, data_mgr_instance)
                
                st.success(f"Updated {category} and rematched suspense transactions!")
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

def render_settings_view(data_mgr, ai_cat, audit_log):
    """Render the settings view with export functionality"""
    st.subheader("⚙️ Settings")

    # Reset options
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️ Reset Data"):
            st.session_state.transactions = pd.DataFrame()
            data_mgr.clear_transactions()
            st.success("All transaction data cleared")

    with col2:
        if st.button("🗑️ Reset Categories"):
            ai_cat.reset_user_categories()  # Only reset user categories, not global AI cache
            st.success("User categories reset")

    # Export functionality
    st.subheader("📤 Export to XML")

    bank_ledger = st.text_input("Bank Ledger Name", value="Bank Account")

    if st.button("Export XML"):
        if not st.session_state.transactions.empty:
            xml_content = generate_xml_export(st.session_state.transactions, bank_ledger)
            st.download_button(
                label="Download XML",
                data=xml_content,
                file_name=f"transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml",
                mime="application/xml"
            )
        else:
            st.warning("No transactions to export")

def generate_xml_export(df, bank_ledger):
    """Generate XML export with ledger information - category is used as software ledger"""
    xml_lines = ['<?xml version="1.0" encoding="UTF-8"?>']
    xml_lines.append('<transactions>')

    for _, row in df.iterrows():
        xml_lines.append('  <transaction>')
        xml_lines.append(f'    <date>{row["date"]}</date>')
        xml_lines.append(f'    <narration><![CDATA[{row["narration"]}]]></narration>')
        xml_lines.append(f'    <amount>{row["amount"]}</amount>')
        xml_lines.append(f'    <category>{row["category"]}</category>')
        xml_lines.append(f'    <bank_ledger>{bank_ledger}</bank_ledger>')
        xml_lines.append(f'    <software_ledger>{row["category"]}</software_ledger>')
        xml_lines.append('  </transaction>')

    xml_lines.append('</transactions>')
    return '\n'.join(xml_lines)

def main():
    """Main application function"""
    # Initialize session state
    init_session_state()

    # Check authentication
    if not st.session_state.authenticated:
        render_login()
        return

    # Initialize components with user-specific data
    (excel_proc, trans_parser, ai_cat, data_mgr, chart_gen, 
     audit_log, rules_engine)= init_components(st.session_state.user_data_dir)

    # Compact header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🏦 Bank Statement Analyzer")
    with col2:
        st.write(f"👤 {st.session_state.username}")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.user_data_dir = None
            st.session_state.transactions = pd.DataFrame()
            st.rerun()

    # Compact navigation
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5, nav_col6 = st.columns(6)

    with nav_col1:
        if st.button("📄 Upload", key="nav_upload"):
            st.session_state.current_view = "Upload & Process"
            # Reset selections when switching tabs
            if 'selected_for_ai_search' in st.session_state:
                del st.session_state.selected_for_ai_search
            if 'ai_search_suspense' in st.session_state:
                del st.session_state.ai_search_suspense
            if 'ai_search_selected' in st.session_state:
                del st.session_state.ai_search_selected
    with nav_col2:
        if st.button("📊 Dashboard", key="nav_dashboard"):
            st.session_state.current_view = "Dashboard"
            # Reset selections when switching tabs
            if 'selected_for_ai_search' in st.session_state:
                del st.session_state.selected_for_ai_search
            if 'ai_search_suspense' in st.session_state:
                del st.session_state.ai_search_suspense
            if 'ai_search_selected' in st.session_state:
                del st.session_state.ai_search_selected
    with nav_col3:
        if st.button("📋 Transactions", key="nav_transactions"):
            st.session_state.current_view = "Transaction Details"
            # Reset selections when switching tabs
            if 'selected_for_ai_search' in st.session_state:
                del st.session_state.selected_for_ai_search
            if 'ai_search_suspense' in st.session_state:
                del st.session_state.ai_search_suspense
            if 'ai_search_selected' in st.session_state:
                del st.session_state.ai_search_selected
    with nav_col4:
        if st.button("🤖 AI Search", key="nav_ai_search"):
            st.session_state.current_view = "AI Search"
            # Reset selections when switching tabs
            if 'selected_for_ai_search' in st.session_state:
                del st.session_state.selected_for_ai_search
            if 'ai_search_suspense' in st.session_state:
                del st.session_state.ai_search_suspense
            if 'ai_search_selected' in st.session_state:
                del st.session_state.ai_search_selected
    with nav_col5:
        if st.button("🏷️ Categories", key="nav_categories"):
            st.session_state.current_view = "Category Management"
            # Reset selections when switching tabs
            if 'selected_for_ai_search' in st.session_state:
                del st.session_state.selected_for_ai_search
            if 'ai_search_suspense' in st.session_state:
                del st.session_state.ai_search_suspense
            if 'ai_search_selected' in st.session_state:
                del st.session_state.ai_search_selected
    with nav_col6:
        if st.button("⚙️ Settings", key="nav_settings"):
            st.session_state.current_view = "Settings"
            # Reset selections when switching tabs
            if 'selected_for_ai_search' in st.session_state:
                del st.session_state.selected_for_ai_search
            if 'ai_search_suspense' in st.session_state:
                del st.session_state.ai_search_suspense
            if 'ai_search_selected' in st.session_state:
                del st.session_state.ai_search_selected

    # Route to views
    if st.session_state.current_view == "Upload & Process":
        render_upload_view(excel_proc, trans_parser, ai_cat, data_mgr, audit_log)
    elif st.session_state.current_view == "Dashboard":
        render_dashboard_view(data_mgr, ai_cat, audit_log)
    elif st.session_state.current_view == "Transaction Details":
        render_transaction_details_view(data_mgr, ai_cat, audit_log)
    elif st.session_state.current_view == "AI Search":
        render_ai_search_view(ai_cat, data_mgr)
    elif st.session_state.current_view == "Category Management":
        render_category_management_view(ai_cat, audit_log)
    elif st.session_state.current_view == "Settings":
        render_settings_view(data_mgr, ai_cat, audit_log)

    # Footer
    st.markdown("---")
    st.markdown("💡 **AI learns from your corrections and gets better over time.**")

if __name__ == "__main__":
    main()