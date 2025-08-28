import pandas as pd
import numpy as np

# -------- 1) Unified loader for CSV / Excel (with optional header detection) --------
def load_table(path, detect_header=True, sheet_name=0, low_memory=False):
    path = str(path)
    is_excel = path.lower().endswith((".xlsx", ".xls"))
    if not detect_header:
        if is_excel:
            return pd.read_excel(path, sheet_name=sheet_name)
        else:
            return pd.read_csv(path, low_memory=low_memory)
    
    # Peek at the first ~20 rows to guess the header row containing '최대' or '최소'
    if is_excel:
        peek = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=20)
    else:
        peek = pd.read_csv(path, header=None, nrows=20, low_memory=low_memory)
    
    header_row_mask = peek.apply(lambda r: r.astype(str).str.contains('최대|최소', na=False).any(), axis=1)
    if header_row_mask.any():
        hdr_idx = header_row_mask.idxmax()  # first True row index
        if is_excel:
            df = pd.read_excel(path, sheet_name=sheet_name, header=hdr_idx)
        else:
            df = pd.read_csv(path, header=hdr_idx, low_memory=low_memory)
    else:
        # fallback: no special header row found; use default behavior
        if is_excel:
            df = pd.read_excel(path, sheet_name=sheet_name)
        else:
            df = pd.read_csv(path, low_memory=low_memory)
    return df

# -------- 2) Utility to find columns by keyword (handles MultiIndex too) --------
def find_columns_with_keyword(columns, keyword: str):
    if isinstance(columns, pd.MultiIndex):
        return [col for col in columns if any(keyword in str(level) for level in col)]
    else:
        return [col for col in columns if keyword in str(col)]

# -------- 3) Your path (Excel) --------
path = r'C:/Users/User/Downloads/한화에어로/압축B작업현황/압출B작업현황일지_0014.xlsx'

df = load_table(path, detect_header=True, sheet_name=0, low_memory=False)

# -------- 4) Find '최대' / '최소' columns and coerce to numeric --------
max_cols = find_columns_with_keyword(df.columns, '최대')
min_cols = find_columns_with_keyword(df.columns, '최소')

if not max_cols and not min_cols:
    print("No columns containing '최대' or '최소' were found. "
          "Check the sheet_name and header detection.")
else:
    print("최대 columns:", max_cols)
    print("최소 columns:", min_cols)

df_num = df.copy()
for c in list(max_cols) + list(min_cols):
    df_num[c] = pd.to_numeric(df_num[c], errors='coerce')

# -------- 5) Build masks and collect indices --------
mask_max = df_num[max_cols].gt(160).any(axis=1) if max_cols else pd.Series(False, index=df.index)
mask_min = df_num[min_cols].gt(160).any(axis=1) if min_cols else pd.Series(False, index=df.index)

idx_max_over_160 = df_num.
