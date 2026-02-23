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

    # --- Feature Engineering ---
    # Create PricePerSqFt 
    df['PricePerSqFt'] = df['SalePrice'] / df['GrLivArea']

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
    
    df['Neighborhood_Full'] = df['Neighborhood'].map(neighborhood_map).fillna(df['Neighborhood'])
    df['Neighborhood'] = df['Neighborhood_Full']

    return df

df = load_data()
market_median_price = df['SalePrice'].median() # Global median for the red line

# ==========================================
# 3. SIDEBAR FILTERS
# ==========================================
st.sidebar.title("🔍 Filter Options")

# 3.1 Price Filter
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
    default=all_neighborhoods
)

# 3.3 Overall Quality Filter
min_qual, max_qual = int(df['OverallQual'].min()), int(df['OverallQual'].max())
selected_quality = st.sidebar.slider("⭐ Select Overall Quality", min_qual, max_qual, (1, 10))

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛋️ Amenities Filters")

# 3.4 Central Air Filter
air_filter = st.sidebar.radio(
    "❄️ Central Air", 
    ["All", "Yes (Y)", "No (N)"], 
    horizontal=True
)

# 3.5 Fireplaces Filter
max_fp = int(df['Fireplaces'].max())
fp_filter = st.sidebar.slider("🔥 Minimum Fireplaces", 0, max_fp, 0)

# 3.6 Garage Filter
max_garage = int(df['GarageCars'].max())
garage_filter = st.sidebar.slider("🚗 Min Garage Capacity (Cars)", 0, max_garage, 0)

# --- FILTERING LOGIC ---
mask = (
    (df['SalePrice'].between(price_range[0], price_range[1])) &
    (df['OverallQual'].between(selected_quality[0], selected_quality[1])) &
    (df['Fireplaces'] >= fp_filter) &
    (df['GarageCars'] >= garage_filter)
)

# Apply Neighborhood Mask
if selected_neighborhoods:
    mask = mask & (df['Neighborhood'].isin(selected_neighborhoods))

# Apply Central Air Mask
if air_filter == "Yes (Y)":
    mask = mask & (df['CentralAir'] == 'Y')
elif air_filter == "No (N)":
    mask = mask & (df['CentralAir'] == 'N')

filtered_df = df[mask]

# ==========================================
# 4. MAIN DASHBOARD UI
# ==========================================
st.title("🏠 Housing Price Prediction & Analytics Dashboard")
st.markdown("### Data Science Semester 2 Project")
st.markdown("---")

# --- Key Metrics Row ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Houses Selected", f"{len(filtered_df)}")
if len(filtered_df) > 0:
    col2.metric("Avg Sale Price", f"${filtered_df['SalePrice'].mean():,.0f}")
    col3.metric("Avg Price per SqFt", f"${filtered_df['PricePerSqFt'].mean():,.2f}")
    col4.metric("Avg Year Built", f"{int(filtered_df['YearBuilt'].mean())}")
else:
    st.error("No data matches your filters.")
    st.stop()

st.markdown("---")

# ==========================================
# GRAPHS SECTION
# ==========================================

# --- ROW 1: Basic EDA ---
c1, c2 = st.columns(2)

with c1:
    st.subheader("📊 Price Distribution")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.histplot(filtered_df['SalePrice'], kde=True, color="skyblue", bins=30, ax=ax1)
    ax1.set_title("Distribution of Sale Prices")
    ax1.set_xlabel("Price ($)")
    st.pyplot(fig1)

with c2:
    st.subheader("🔥 Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    corr_cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']
    corr_matrix = filtered_df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
    ax2.set_title("Correlation Matrix")
    st.pyplot(fig2)

st.markdown("---")

# --- ROW 2: Year/Quality & Amenities ---
c3, c4 = st.columns(2)

with c3:
    st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
    st.subheader("💡 Impact of Year Built & Quality on Price")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=filtered_df, x='YearBuilt', y='SalePrice', hue='OverallQual', palette='RdYlGn', alpha=0.7, ax=ax3)
    ax3.set_title('Impact of Year Built and Overall Quality on Sale Price')
    st.pyplot(fig3)

with c4:
    st.subheader("💡 Impact of Key Amenities on Sale Price")
    # Add a selectbox for the user to choose which amenity to view
    amenity_choice = st.selectbox(
        "Select Amenity to Analyze:", 
        options=['CentralAir', 'Fireplaces', 'GarageCars'],
        format_func=lambda x: 'Central Air' if x == 'CentralAir' else ('Fireplaces' if x == 'Fireplaces' else 'Garage Capacity')
    )
    
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=amenity_choice, y='SalePrice', data=filtered_df, palette='Set2', ax=ax4)
    # Add horizontal line for the global market median price
    ax4.axhline(market_median_price, color='red', linestyle='--', label=f'Market Median (${market_median_price:,.0f})')
    ax4.set_title(f'{amenity_choice} vs. Sale Price')
    ax4.legend()
    st.pyplot(fig4)

st.markdown("---")

# --- ROW 3: Neighborhood (Full Width) ---
st.subheader("💡 Neighborhood Analysis")

# Add a selector for the metric
neighborhood_metric = st.radio(
    "Select Metric to Visualize:",
    options=["Average Price per SqFt (Best Value)", "Average Total Sale Price"],
    horizontal=True
)

fig5, ax5 = plt.subplots(figsize=(16, 6)) # Wider figure for full row

if neighborhood_metric == "Average Price per SqFt (Best Value)":
    # Calculate mean price per sqft and sort ascending (Lowest Price/SqFt = Best Value)
    val_hood = filtered_df.groupby('Neighborhood')['PricePerSqFt'].mean().sort_values()
    sns.barplot(x=val_hood.index, y=val_hood.values, palette='viridis', ax=ax5)
    ax5.set_title('Average Price per SqFt by Neighborhood (Sorted by Best Value)')
    ax5.set_ylabel('Price per SqFt ($)')

else:
    # Calculate mean sale price and sort descending (Highest Price first)
    val_hood = filtered_df.groupby('Neighborhood')['SalePrice'].mean().sort_values(ascending=False)
    sns.barplot(x=val_hood.index, y=val_hood.values, palette='magma', ax=ax5)
    ax5.set_title('Average Sale Price by Neighborhood (Sorted by Highest Price)')
    ax5.set_ylabel('Average Sale Price ($)')

# Apply formatting to whichever graph is selected
plt.xticks(rotation=45, ha='right')
ax5.set_xlabel('Neighborhood')
st.pyplot(fig5)

st.markdown("---")

# ==========================================
# 5. MACHINE LEARNING SECTION
# ==========================================
st.subheader("🤖 Machine Learning Prediction (Random Forest)")

col_ml1, col_ml2 = st.columns([1, 2])

with col_ml1:
    st.markdown("""
    **Model Information:**
    - **Algorithm:** Random Forest Regressor
    - **Features Used:** Overall Quality, Living Area, Garage Capacity, Total Basement SF, Year Built.
    - **Purpose:** To predict housing prices based on property characteristics and evaluate the model's accuracy.
    """)

with col_ml2:
    if len(filtered_df) > 10:
        # Prepare Data
        features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']
        X = filtered_df[features]
        y = filtered_df['SalePrice']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Model
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        # Plot
        fig6, ax6 = plt.subplots(figsize=(8, 4))
        sns.scatterplot(x=y_test, y=y_pred, color="purple", alpha=0.6, ax=ax6)
        
        # Perfect prediction line
        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax6.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction")
        
        ax6.set_xlabel("Actual Price ($)")
        ax6.set_ylabel("Predicted Price ($)")
        ax6.set_title(f"Actual vs Predicted Prices (Accuracy R²: {r2_score(y_test, y_pred):.2f})")
        ax6.legend()
        st.pyplot(fig6)
    else:
        st.warning("Not enough data points selected to run prediction model. Please adjust your filters.")

# ==========================================
# 6. RAW DATA
# ==========================================
if st.checkbox("Show Raw Data"):
    st.dataframe(filtered_df)