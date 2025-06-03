import streamlit as st
import pandas as pd
import numpy as np

st.title("🔍 Debug Data Loading")
st.write("Let's see exactly what's in your Excel file and why the actual data isn't showing up.")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

if uploaded_file:
    try:
        # Load raw data
        raw_df = pd.read_excel(uploaded_file)
        
        st.subheader("1. Raw Excel Data")
        st.write(f"Shape: {raw_df.shape}")
        st.write("Column names:")
        st.write(list(raw_df.columns))
        st.dataframe(raw_df.head(20))
        
        # Check if required columns exist
        st.subheader("2. Column Analysis")
        if 'Month' in raw_df.columns:
            st.success("✅ 'Month' column found")
        else:
            st.error("❌ 'Month' column missing")
            st.write("Available columns:", list(raw_df.columns))
            
        if 'Sales' in raw_df.columns:
            st.success("✅ 'Sales' column found")
        else:
            st.error("❌ 'Sales' column missing")
            st.write("Available columns:", list(raw_df.columns))
        
        if 'Month' in raw_df.columns and 'Sales' in raw_df.columns:
            # Convert Month to datetime
            df_clean = raw_df.copy()
            df_clean['Month'] = pd.to_datetime(df_clean['Month'], errors='coerce')
            df_clean['Sales'] = pd.to_numeric(df_clean['Sales'], errors='coerce')
            
            # Remove invalid data
            df_clean = df_clean.dropna(subset=['Month', 'Sales'])
            
            st.subheader("3. Cleaned Data")
            st.write(f"Shape after cleaning: {df_clean.shape}")
            st.dataframe(df_clean.head(20))
            
            # Aggregate by month
            monthly = df_clean.groupby('Month', as_index=False)['Sales'].sum().sort_values('Month')
            
            st.subheader("4. Monthly Aggregated Data")
            st.write(f"Shape: {monthly.shape}")
            st.dataframe(monthly)
            
            # Show date range
            st.subheader("5. Date Range Analysis")
            st.write(f"Date range: {monthly['Month'].min()} to {monthly['Month'].max()}")
            
            # Check for 2024 data specifically
            data_2024 = monthly[
                (monthly['Month'] >= pd.Timestamp('2024-01-01')) & 
                (monthly['Month'] < pd.Timestamp('2025-01-01'))
            ]
            
            st.subheader("6. 2024 Data Check")
            if len(data_2024) > 0:
                st.success(f"✅ Found {len(data_2024)} months of 2024 data!")
                st.dataframe(data_2024)
                st.write(f"Total 2024 sales: {data_2024['Sales'].sum():,.2f}")
            else:
                st.error("❌ No 2024 data found")
                
                # Show what years we do have
                monthly['Year'] = monthly['Month'].dt.year
                years_available = sorted(monthly['Year'].unique())
                st.write(f"Available years: {years_available}")
                
                # Show monthly data by year
                for year in years_available:
                    year_data = monthly[monthly['Year'] == year]
                    st.write(f"**{year}:** {len(year_data)} months, Total sales: {year_data['Sales'].sum():,.2f}")
            
            # Check training data (through 2023)
            train_data = monthly[monthly['Month'] <= pd.Timestamp('2023-12-01')]
            st.subheader("7. Training Data Check")
            if len(train_data) > 0:
                st.success(f"✅ Found {len(train_data)} months of training data through 2023")
                st.write(f"Training range: {train_data['Month'].min()} to {train_data['Month'].max()}")
            else:
                st.error("❌ No training data found through 2023")
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.write("Please check your file format and try again.")
