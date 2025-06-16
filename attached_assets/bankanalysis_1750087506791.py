import pandas as pd
import re

# 🏦 Input bank ledger
bank_ledger = input("Enter the Bank Ledger Name (e.g., HDFC Bank A/c): \n\n").strip()

suspense_led = input("Enter the Suspense Ledger Name (e.g., Suspense A/c): \n\n").strip()

# 🧼 Cleaning for matching only
def extract_clean_name(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[\\/|@#:,]', ' ', text)
    text = re.sub(r'\b(upi|paytm|gpay|phonepe|to|from|txn|transaction|ref|Ch. No. :)\b', '', text)
    text = re.sub(r'\d{5,}', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().title()

# 🔁 Similarity function
def is_name_similar(name1, name2):
    if pd.isna(name1) or pd.isna(name2):
        return 0.0
    name1 = extract_clean_name(name1)
    name2 = extract_clean_name(name2)
    tokens1 = set(name1.split())
    tokens2 = set(name2.split())
    if not tokens1 or not tokens2:
        return 0.0
    return round(len(tokens1 & tokens2) / max(len(tokens1), len(tokens2)), 2)

# 📥 Load Excel
master_df = pd.read_excel('Book1.xlsx', sheet_name='Master')
bkst_df = pd.read_excel('Book1.xlsx', sheet_name='bkst')

# 📋 Normalize column names
master_df.columns = master_df.columns.str.strip().str.lower()
bkst_df.columns = bkst_df.columns.str.strip().str.lower()

# 🧮 Fill amounts if missing
master_df['low'] = master_df['low'].fillna(0)
master_df['high'] = master_df['high'].fillna(99999999)

# 🧹 Clean string cells
master_df = master_df.applymap(lambda x: x.lower().strip() if isinstance(x, str) else x)
bkst_df = bkst_df.applymap(lambda x: x.lower().strip() if isinstance(x, str) else x)

# 🧽 Extract cleaned description for internal use
bkst_df['cleaned_description'] = bkst_df['description'].apply(extract_clean_name)
master_df['cleaned_partic'] = master_df['partic'].apply(extract_clean_name)



# 🔁 Map Dr/Cr to type
def map_type(drcr):
    return 'payment' if drcr == 'dr' else 'receipt'
bkst_df['mapped_type'] = bkst_df['dr / cr'].apply(map_type)

# 📤 Final output
export_rows = []

# 🔍 Loop through transactions
for _, bk_row in bkst_df.iterrows():
    raw_desc = bk_row.get('description')
    cleaned_desc = bk_row.get('cleaned_description')
    bk_amt = bk_row.get('amount')
    bk_type = bk_row.get('mapped_type')
    bk_drcr = bk_row.get('dr / cr')
    bk_date = bk_row.get('value date')

    if pd.isna(raw_desc) or pd.isna(bk_amt) or pd.isna(bk_type):
        continue

    matched = False
    best_match = {'similarity': 0, 'ledger': f'{suspense_led}'}

    for _, master_row in master_df.iterrows():
        master_partic = master_row.get('cleaned_partic')
        master_low = master_row.get('low', 0)
        master_high = master_row.get('high', 99999999)
        master_type = master_row.get('type')
        master_ledger = master_row.get('ledger')

        #if pd.isna(master_partic) or pd.isna(master_type):
        #    continue

        similarity = is_name_similar(cleaned_desc, master_partic)
        amount_in_range = master_low <= bk_amt <= master_high
        #type_matches = bk_type and type_matches
        print(f"🔍 Trying: {cleaned_desc} ↔ {master_partic} → Sim: {similarity}, InRange: {amount_in_range}")

        if similarity >= 0.2 and amount_in_range and similarity > best_match['similarity']:
            matched = True
            best_match = {'similarity': similarity, 'ledger': master_ledger}
        

    # 💳 Construct journal entry
    if bk_drcr == 'cr':
        vouchertype = "Receipt"
        cr_ledger = best_match['ledger']
        dr_ledger = bank_ledger
        dr_amt = bk_amt
        cr_amt = bk_amt
    else:
        vouchertype = "Payment"
        cr_ledger = bank_ledger
        dr_ledger = best_match['ledger']
        dr_amt = bk_amt
        cr_amt = bk_amt

    export_rows.append({
        'Date': bk_date,
        'Narration': raw_desc,  # original uncleaned
        'VchType': vouchertype,
        'DebitLedger': dr_ledger,
        'DrAmount': dr_amt,
        'CreditLedger': cr_ledger,
        'CrAmount': cr_amt,
        'Similarity': best_match['similarity'] if matched else 0
    })

# 🧾 Export
export_df = pd.DataFrame(export_rows)
export_df.to_excel('Exported_Ledgers_With_Similarity.xlsx', index=False)

print("✅ Exported with similarity scores. Check 'Exported_Ledgers_With_Similarity.xlsx'")
