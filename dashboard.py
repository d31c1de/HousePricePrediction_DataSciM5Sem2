import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="MWIT Data Science Final Project",
    page_icon="🏠",
    layout="wide"
)

# ==========================================
# 2. DATA LOADING & MAPPING
# ==========================================
@st.cache_data
def load_data():
    file_path = 'data/train.csv'
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File not found at: {file_path}")
        st.stop()

    # Basic Cleaning for Visualization
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna('None')

    # --- MAP NEIGHBORHOOD NAMES ---
    neighborhood_map = {
        "Blmngtn": "Bloomington Heights",
        "Blueste": "Bluestem",
        "BrDale": "Briardale",
        "BrkSide": "Brookside",
        "ClearCr": "Clear Creek",
        "CollgCr": "College Creek",
        "Crawfor": "Crawford",
        "Edwards": "Edwards",
        "Gilbert": "Gilbert",
        "IDOTRR": "Iowa DOT and Rail Road",
        "MeadowV": "Meadow Village",
        "Mitchel": "Mitchell",
        "NAmes": "North Ames", 
        "NoRidge": "Northridge",
        "NPkVill": "Northpark Villa",
        "NridgHt": "Northridge Heights",
        "NWAmes": "Northwest Ames",
        "OldTown": "Old Town",
        "SWISU": "South & West of Iowa State U",
        "Sawyer": "Sawyer",
        "SawyerW": "Sawyer West",
        "Somerst": "Somerset",
        "StoneBr": "Stone Brook",
        "Timber": "Timberland",
        "Veenker": "Veenker"
    }
    
    # Apply mapping. If a code isn't found, keep original.
    df['Neighborhood_Full'] = df['Neighborhood'].map(neighborhood_map).fillna(df['Neighborhood'])
    
    # Use the full name for the 'Neighborhood' column for display purposes
    df['Neighborhood'] = df['Neighborhood_Full']

    return df

df = load_data()

# ==========================================
# 3. SIDEBAR FILTERS
# ==========================================
st.sidebar.title("🔍 Filter Options")

# 3.1 Price Filter (Added)
min_p, max_p = int(df['SalePrice'].min()), int(df['SalePrice'].max())
price_range = st.sidebar.slider(
    "💰 Select Price Range ($)", 
    min_p, max_p, 
    (min_p, max_p)
)

# 3.2 Neighborhood Filter
all_neighborhoods = sorted(df['Neighborhood'].unique())
selected_neighborhoods = st.sidebar.multiselect(
    "📍 Select Neighborhood(s)", 
    all_neighborhoods, 
    default=all_neighborhoods[:3] 
)

# 3.3 Overall Quality Filter
min_qual, max_qual = int(df['OverallQual'].min()), int(df['OverallQual'].max())
selected_quality = st.sidebar.slider("⭐ Select Overall Quality", min_qual, max_qual, (4, 10))

# Filtering Logic
mask = (
    (df['SalePrice'].between(price_range[0], price_range[1])) &
    (df['OverallQual'].between(selected_quality[0], selected_quality[1]))
)

if selected_neighborhoods:
    mask = mask & (df['Neighborhood'].isin(selected_neighborhoods))

filtered_df = df[mask]

# ==========================================
# 4. MAIN DASHBOARD UI
# ==========================================
st.title("🏠 Housing Price Prediction & Analytics Dashboard")
st.markdown("### Data Science Semester 2 Project")
st.markdown("---")

# --- Key Metrics Row ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Houses", f"{len(filtered_df)}")
if len(filtered_df) > 0:
    col2.metric("Avg Sale Price", f"${filtered_df['SalePrice'].mean():,.0f}")
    col3.metric("Avg Living Area", f"{filtered_df['GrLivArea'].mean():,.0f} sqft")
    col4.metric("Avg Year Built", f"{int(filtered_df['YearBuilt'].mean())}")
else:
    st.error("No data matches your filters.")
    st.stop()

st.markdown("---")

# --- ROW 1: Distribution & Correlation (Graphs 1 & 2) ---
c1, c2 = st.columns(2)

with c1:
    st.subheader("📊 1. Price Distribution")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.histplot(filtered_df['SalePrice'], kde=True, color="skyblue", bins=30, ax=ax1)
    ax1.set_title("Distribution of Sale Prices (Filtered)")
    ax1.set_xlabel("Price ($)")
    st.pyplot(fig1)

with c2:
    st.subheader("🔥 2. Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    corr_cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']
    corr_matrix = filtered_df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
    ax2.set_title("Correlation Matrix")
    st.pyplot(fig2)

# --- ROW 2: Living Area & Quality (Graphs 3 & 4) ---
c3, c4 = st.columns(2)

with c3:
    st.subheader("📐 3. Living Area vs. Price")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=filtered_df, x='GrLivArea', y='SalePrice', hue='OverallQual', palette='viridis', ax=ax3)
    ax3.set_title("Price vs. Square Footage")
    st.pyplot(fig3)

with c4:
    st.subheader("⭐ 4. Impact of Quality on Price")
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=filtered_df, x='OverallQual', y='SalePrice', palette="magma", ax=ax4)
    ax4.set_title("Sale Price Range by Overall Quality")
    st.pyplot(fig4)

# --- ROW 3: Neighborhood & Age (Graphs 5 & 6) ---
st.subheader("📍 5. Neighborhood Pricing")
fig5, ax5 = plt.subplots(figsize=(12, 5))
# Calculate order by median price
order = filtered_df.groupby("Neighborhood")["SalePrice"].median().sort_values(ascending=False).index
sns.barplot(data=filtered_df, x='Neighborhood', y='SalePrice', order=order, palette="Blues_r", errorbar=None, ax=ax5)
plt.xticks(rotation=45)
ax5.set_title("Average Price per Neighborhood")
st.pyplot(fig5)

c5, c6 = st.columns(2)
with c5:
    st.subheader("📅 6. Year Built vs. Price")
    fig6, ax6 = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=filtered_df, x='YearBuilt', y='SalePrice', color="green", ax=ax6)
    ax6.set_title("Price Trend by Year Built")
    st.pyplot(fig6)

# ==========================================
# 5. MACHINE LEARNING SECTION (Graph 7)
# ==========================================
with c6:
    st.subheader("🤖 7. Model Prediction (Actual vs Predicted)")
    
    if len(filtered_df) > 10:
        # Prepare Data
        X = filtered_df[['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']]
        y = filtered_df['SalePrice']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Simple Random Forest
        rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        # Plot
        fig7, ax7 = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x=y_test, y=y_pred, color="purple", alpha=0.6, ax=ax7)
        
        # Perfect line
        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax7.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Prediction")
        
        ax7.set_xlabel("Actual Price")
        ax7.set_ylabel("Predicted Price")
        ax7.set_title(f"Model Accuracy (R²: {r2_score(y_test, y_pred):.2f})")
        ax7.legend()
        st.pyplot(fig7)
    else:
        st.warning("Not enough data points selected to run prediction model.")

# ==========================================
# 6. RAW DATA
# ==========================================
if st.checkbox("Show Raw Data"):
    st.dataframe(filtered_df)