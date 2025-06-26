import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import calendar
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Walmart Sales Analysis & Forecasting", page_icon="📈", layout="wide")

# Walmart theme colors and styling
st.markdown("""<style>
.main-header {
    background: linear-gradient(135deg, #004c91 0%, #0071ce 50%, #ffc220 100%);
    color: white; padding: 2rem; border-radius: 1rem; text-align: center; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.stButton > button {
    background: linear-gradient(135deg, #0071ce 0%, #004c91 100%); color: white; border: none; border-radius: 0.5rem; 
    padding: 0.75rem 2rem; font-weight: 600; transition: all 0.3s ease;
}
.stButton > button:hover { background: linear-gradient(135deg, #004c91 0%, #0071ce 100%); transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); }

div.stAlert[data-baseweb="notification"] > div:first-child { background-color: #d4edda !important; border-left: 4px solid #28a745 !important; color: #155724 !important; }
div[data-testid="stAlert"] > div:first-child { background-color: #d4edda !important; border-left: 4px solid #28a745 !important; color: #155724 !important; }

div.stAlert[data-baseweb="notification"]:has([data-testid="stMarkdownContainer"]:contains("⚠️")) > div:first-child,
div.stAlert[data-baseweb="notification"]:has([data-testid="stMarkdownContainer"]:contains("❌")) > div:first-child,
div[data-testid="stAlert"]:has([data-testid="stMarkdownContainer"]:contains("⚠️")) > div:first-child,
div[data-testid="stAlert"]:has([data-testid="stMarkdownContainer"]:contains("❌")) > div:first-child { 
    background-color: #f8d7da !important; border-left: 4px solid #dc3545 !important; color: #721c24 !important; 
}
</style>""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1>🛒 Walmart Sales Analysis & Forecasting</h1></div>', unsafe_allow_html=True)

# Navigation buttons
st.markdown("### Navigation")
col1, col2, col3 = st.columns([4, 4, 4])

with col1:
    if st.button("📋 Overview", key="overview_btn", use_container_width=True):
        st.session_state.page = "overview"

with col2:
    if st.button("📈 Data Analysis", key="analysis_btn", use_container_width=True):
        st.session_state.page = "analysis"

with col3:
    if st.button("🔮 Forecasting", key="forecasting_btn", use_container_width=True):
        st.session_state.page = "forecasting"

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "overview"

# Get the current page
page = st.session_state.page

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('dataset.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').set_index('Date')
        df = df[df['Dept'] == 23].drop(columns=['Dept'])
        return df
    except FileNotFoundError:
        st.error("Dataset not found!")
        return None

@st.cache_resource
def load_ann_model():
    try:
        custom_objects = {'mse': tf.keras.metrics.MeanSquaredError(), 'mean_squared_error': tf.keras.metrics.MeanSquaredError()}
        model = load_model('model.h5', custom_objects=custom_objects, compile=False)
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        scaler = joblib.load('scaler_dl.pkl')
        return model, scaler
    except Exception as e:
        st.warning(f"Model loading error: {str(e)}")
        return None, None

df = load_data()

if df is not None:
    if page == "overview":
        st.markdown("---")
        st.header("📋 Dataset Overview")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("📊 Total Records", f"{len(df):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.metric("📅 Start Date", df.index.min().strftime('%Y-%m-%d'))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.metric("📅 End Date", df.index.max().strftime('%Y-%m-%d'))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.metric("💰 Average Sales", f"${df['Weekly_Sales'].mean():,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col5:
            st.metric("📈 Max Sales", f"${df['Weekly_Sales'].max():,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col6:
            st.metric("📉 Min Sales", f"${df['Weekly_Sales'].min():,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
    elif page == "analysis":
        st.markdown("---")
        st.header("📈 Data Analysis")
        
        # Weekly Sales Over Time
        st.subheader("📈 Weekly Sales Over Time")
        max_idx, min_idx = df['Weekly_Sales'].idxmax(), df['Weekly_Sales'].idxmin()
        max_val, min_val = df.loc[max_idx, 'Weekly_Sales'], df.loc[min_idx, 'Weekly_Sales']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Weekly_Sales'], mode='lines', name='Weekly Sales', line=dict(color='#0071ce', width=2)))
        fig.add_trace(go.Scatter(x=[max_idx], y=[max_val], mode='markers+text', name=f'Max: ${max_val:,.0f}', marker=dict(color='#28a745', size=12, line=dict(color='#004c91', width=2)), text=[f'Max: ${max_val:,.0f}'], textposition='top center'))
        fig.add_trace(go.Scatter(x=[min_idx], y=[min_val], mode='markers+text', name=f'Min: ${min_val:,.0f}', marker=dict(color='#dc3545', size=12, line=dict(color='#004c91', width=2)), text=[f'Min: ${min_val:,.0f}'], textposition='bottom center'))
        fig.update_layout(title="Weekly Sales Over Time", xaxis_title="Date", yaxis_title="Sales ($)", height=400, showlegend=True, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Sales Distribution
        st.subheader("📊 Sales Distribution")
        fig = px.histogram(df, x='Weekly_Sales', nbins=50, title="Distribution of Weekly Sales", color_discrete_sequence=['#0071ce'], marginal='box')
        fig.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Holiday Analysis
        st.subheader("🎉 Holiday Impact Analysis")
        col1, col2 = st.columns(2)
        with col1:
            holiday_counts = df['IsHoliday'].value_counts()
            fig = px.bar(x=['Non-Holiday', 'Holiday'], y=holiday_counts.values, title="Count of Holiday vs Non-Holiday Weeks", color=['Non-Holiday', 'Holiday'], color_discrete_sequence=['#28a745', '#dc3545'], text=holiday_counts.values)
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            holiday_avg = df.groupby('IsHoliday')['Weekly_Sales'].mean()
            fig = px.bar(x=['Non-Holiday', 'Holiday'], y=holiday_avg.values, title="Average Weekly Sales", color=['Non-Holiday', 'Holiday'], color_discrete_sequence=['#28a745', '#dc3545'], text=[f'${val:,.0f}' for val in holiday_avg.values])
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        # Monthly Pattern
        st.subheader("📅 Monthly Sales Pattern")
        monthly_sales = df.groupby(df.index.month)['Weekly_Sales'].mean()
        month_names = [calendar.month_abbr[m] for m in monthly_sales.index]   

        max_month_idx = monthly_sales.idxmax()
        min_month_idx = monthly_sales.idxmin()
        max_month_val = monthly_sales.max()
        min_month_val = monthly_sales.min()
        max_month_name = calendar.month_abbr[max_month_idx]
        min_month_name = calendar.month_abbr[min_month_idx]
    
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=month_names, y=monthly_sales.values, mode='lines+markers+text', name='Average Sales', line=dict(color='#0071ce', width=3), marker=dict(size=10, color='#ffc220', line=dict(color='#004c91', width=2)), text=[f'${val:,.0f}' for val in monthly_sales.values], textposition='top center'))        
        fig.add_trace(go.Scatter(x=[max_month_name], y=[max_month_val], mode='markers+text', name=f'Max: {max_month_name} (${max_month_val:,.0f})', marker=dict(color='#28a745', size=15, line=dict(color='#004c91', width=3)), text=[f'Max: ${max_month_val:,.0f}'], textposition='top center', textfont=dict(color='#28a745', size=12)))        
        fig.add_trace(go.Scatter(x=[min_month_name], y=[min_month_val], mode='markers+text', name=f'Min: {min_month_name} (${min_month_val:,.0f})', marker=dict(color='#dc3545', size=15, line=dict(color='#004c91', width=3)), text=[f'Min: ${min_month_val:,.0f}'], textposition='bottom center', textfont=dict(color='#dc3545', size=12)))
        fig.update_layout(title="Monthly Sales Pattern", xaxis_title="Month", yaxis_title="Average Sales ($)", height=400, showlegend=True, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation Matrix
        st.subheader("🔗 Correlation Analysis")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Matrix", color_continuous_scale='rdbu', zmin=-1, zmax=1)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Time Series Analysis
        st.header("🔍 Time Series Analysis")
        
        # Stationarity Test
        st.subheader("📊 Stationarity Test")
        adf_result = adfuller(df['Weekly_Sales'].dropna())
        col1, col2 = st.columns(2)
        with col1: 
            st.metric("ADF Test Statistic", f"{adf_result[0]:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2: 
            st.metric("P-value", f"{adf_result[1]:.6f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if adf_result[1] <= 0.05:
            st.success("✅ Series is stationary (p-value ≤ 0.05)")
        else:
            st.warning("⚠️ Series is non-stationary (p-value > 0.05)")
        
        # Seasonal Decomposition
        st.subheader("📈 Seasonal Decomposition")
        try:
            decomposition = seasonal_decompose(df['Weekly_Sales'], model='additive', period=52)
            fig = make_subplots(rows=4, cols=1, subplot_titles=['Original Series', 'Trend Component', 'Seasonal Component', 'Residual Component'], vertical_spacing=0.05)
            
            fig.add_trace(go.Scatter(x=df.index, y=decomposition.observed, mode='lines', name='Original', line=dict(color='#0071ce', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=decomposition.trend, mode='lines', name='Trend', line=dict(color='#ffc220', width=2)), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=decomposition.seasonal, mode='lines', name='Seasonal', line=dict(color="#28a745", width=1.5)), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=decomposition.resid, mode='lines', name='Residual', line=dict(color='#dc3545', width=1)), row=4, col=1)
            
            fig.update_layout(height=1200, showlegend=False, title_text="Time Series Decomposition", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error in decomposition: {str(e)}")
        
        # ACF and PACF
        st.subheader("📊 Autocorrelation Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Autocorrelation Function (ACF)**")
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_acf(df['Weekly_Sales'].dropna(), ax=ax, lags=40, title='ACF', color='#0071ce')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.write("**Partial Autocorrelation Function (PACF)**")
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_pacf(df['Weekly_Sales'].dropna(), ax=ax, lags=40, title='PACF', color='#0071ce')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    elif page == "forecasting":
        st.markdown("---")
        st.header("🎯 Sales Forecasting")
        
        model, scaler = load_ann_model()
        
        if model is not None and scaler is not None:
            st.success("✅ Model loaded successfully!")

            st.subheader("⚙️ Forecast Settings")
            forecast_periods = st.slider("Number of weeks to forecast:", min_value=1, max_value=13, value=7)
            
            if st.button("🔮 Generate Forecast", type="primary"):
                try:
                    lookback = 13
                    scaled_data = scaler.transform(df[['Weekly_Sales']])
                    last_sequence = scaled_data[-lookback:, 0].reshape(1, -1)
                    forecasts = []
                    current_sequence = last_sequence.copy()
                    
                    for i in range(forecast_periods):
                        next_pred = model.predict(current_sequence, verbose=0)
                        forecasts.append(next_pred[0, 0])
                        current_sequence = np.roll(current_sequence, -1)
                        current_sequence[0, -1] = next_pred[0, 0]
                    
                    forecast_original = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
                    last_date = df.index[-1]
                    forecast_dates = pd.date_range(start=last_date + timedelta(days=7), periods=forecast_periods, freq='W')
                    
                    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast_original})
                    
                    st.subheader("📈 Forecast Results")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1: 
                        st.metric("Average Forecast", f"${forecast_original.mean():,.0f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2: 
                        st.metric("Max Forecast", f"${forecast_original.max():,.0f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col3: 
                        st.metric("Min Forecast", f"${forecast_original.min():,.0f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col4: 
                        st.metric("Total Forecast", f"${forecast_original.sum():,.0f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index[-52:], y=df['Weekly_Sales'][-52:], mode='lines', name='Historical', line=dict(color='#0071ce', width=2)))
                    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_original, mode='lines+markers', name='Forecast', line=dict(color='#ffc220', width=3), marker=dict(size=10, color='#ffc220', line=dict(color='#004c91', width=2))))
                    fig.update_layout(title="Sales Forecast", xaxis_title="Date", yaxis_title="Sales ($)", height=500, hovermode='x unified', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("📊 Detailed Forecast")
                    forecast_display = forecast_df.copy()
                    forecast_display['Forecast'] = forecast_display['Forecast'].apply(lambda x: f"${x:,.0f}")
                    st.dataframe(forecast_display, use_container_width=True, hide_index=True)
                    
                    csv = forecast_df.to_csv(index=False)
                    st.download_button("📥 Download Forecast as CSV", data=csv, file_name=f"sales_forecast_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
                    
                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")
        
        else:
            st.error("❌ Could not load the model or scaler.")
            st.subheader("⚙️ Forecast Settings")
            st.slider("Number of weeks to forecast:", 1, 13, 7, disabled=True)
            st.button("🔮 Generate Forecast", type="primary", disabled=True)

else:
    st.error("❌ Unable to load dataset. Please ensure 'dataset.csv' is available in the app directory.")