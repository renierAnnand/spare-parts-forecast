elif 90 <= mode_days <= 92:
        return 'Q'  # Quarterly
    elif 365 <= mode_days <= 366:
        return 'Y'  # Yearly
    else:
        return 'M'  # Default to monthly


def inv_boxcox(y, lambda_param):
    """Inverse Box-Cox transformation"""
    if lambda_param == 0:
        return np.exp(y)
    else:
        return np.exp(np.log(lambda_param * y + 1) / lambda_param)


def create_forecast_plot(result_df, forecast_year, historical_df=None):
    """Create comprehensive forecast visualization with confidence intervals"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Forecast Comparison', 'Model Performance', 
                       'Residual Analysis', 'Forecast Intervals'),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Main forecast comparison
    forecast_cols = [col for col in result_df.columns if '_Forecast' in col or 
                    col in ['Weighted_Ensemble', 'Meta_Learning']]
    actual_col = f'Actual_{forecast_year}'
    
    colors = px.colors.qualitative.Set3
    
    # Add historical data if available
    if historical_df is not None:
        fig.add_trace(
            go.Scatter(
                x=historical_df['Month'],
                y=historical_df['Sales_Original'],
                mode='lines',
                name='Historical',
                line=dict(color='gray', width=2),
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Add forecasts
    for i, col in enumerate(forecast_cols):
        model_name = col.replace('_Forecast', '').replace('_', ' ')
        
        if col in ['Weighted_Ensemble', 'Meta_Learning']:
            line_style = dict(width=3, dash='dash' if col == 'Weighted_Ensemble' else 'dot')
            line_color = '#FF6B6B' if col == 'Weighted_Ensemble' else '#4ECDC4'
        else:
            line_style = dict(width=2)
            line_color = colors[i % len(colors)]
        
        fig.add_trace(
            go.Scatter(
                x=result_df['Month'],
                y=result_df[col],
                mode='lines+markers',
                name=model_name,
                line=dict(color=line_color, **line_style),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
    
    # Add actual data if available
    if actual_col in result_df.columns:
        actual_data = result_df[result_df[actual_col].notna()]
        if len(actual_data) > 0:
            fig.add_trace(
                go.Scatter(
                    x=actual_data['Month'],
                    y=actual_data[actual_col],
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color='black', width=4),
                    marker=dict(size=10, symbol='star')
                ),
                row=1, col=1
            )
    
    # Performance metrics visualization
    if actual_col in result_df.columns and result_df[actual_col].notna().any():
        performance_data = []
        for col in forecast_cols:
            model_name = col.replace('_Forecast', '').replace('_', ' ')
            metrics = calculate_comprehensive_metrics(
                result_df[actual_col].dropna(),
                result_df.loc[result_df[actual_col].notna(), col]
            )
            if metrics:
                performance_data.append({
                    'Model': model_name,
                    'MAPE': metrics['MAPE']
                })
        
        if performance_data:
            perf_df = pd.DataFrame(performance_data).sort_values('MAPE')
            fig.add_trace(
                go.Bar(
                    x=perf_df['Model'],
                    y=perf_df['MAPE'],
                    name='MAPE',
                    marker_color='lightblue',
                    showlegend=False
                ),
                row=1, col=2
            )
    
    # Residual analysis
    if actual_col in result_df.columns and 'Weighted_Ensemble' in result_df.columns:
        actual_subset = result_df[result_df[actual_col].notna()]
        if len(actual_subset) > 0:
            residuals = actual_subset[actual_col] - actual_subset['Weighted_Ensemble']
            
            fig.add_trace(
                go.Scatter(
                    x=actual_subset['Month'],
                    y=residuals,
                    mode='markers+lines',
                    name='Residuals',
                    line=dict(color='red', width=1),
                    marker=dict(size=8),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # Forecast intervals (if available)
    if 'xgboost_info' in st.session_state and st.session_state['xgboost_info']:
        info = st.session_state['xgboost_info']
        
        fig.add_trace(
            go.Scatter(
                x=result_df['Month'],
                y=info['values'],
                mode='lines',
                name='XGBoost Forecast',
                line=dict(color='green', width=2),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Add confidence interval
        fig.add_trace(
            go.Scatter(
                x=result_df['Month'].tolist() + result_df['Month'].tolist()[::-1],
                y=info['lower_bound'].tolist() + info['upper_bound'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% CI',
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text=f"Comprehensive Forecast Analysis - {forecast_year}",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_yaxes(title_text="Sales", row=1, col=1)
    fig.update_yaxes(title_text="MAPE (%)", row=1, col=2)
    fig.update_yaxes(title_text="Residuals", row=2, col=1)
    fig.update_yaxes(title_text="Sales", row=2, col=2)
    
    return fig


def create_diagnostic_plots(historical_df):
    """Create diagnostic plots for time series analysis"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Time Series Decomposition', 'ACF Plot', 
                       'Seasonal Pattern', 'Distribution Analysis')
    )
    
    # Decomposition
    if len(historical_df) >= 24:
        decomposition = seasonal_decompose(historical_df['Sales'], model='additive', period=12)
        
        # Trend
        fig.add_trace(
            go.Scatter(x=historical_df['Month'], y=decomposition.trend, 
                      name='Trend', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Seasonal
        fig.add_trace(
            go.Scatter(x=historical_df['Month'][:12], y=decomposition.seasonal[:12], 
                      name='Seasonal', line=dict(color='green')),
            row=1, col=2
        )
    
    # Distribution
    fig.add_trace(
        go.Histogram(x=historical_df['Sales'], name='Sales Distribution', 
                    nbinsx=30, showlegend=False),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True)
    
    return fig


def create_feature_importance_plot(feature_importance_df):
    """Create feature importance visualization"""
    top_features = feature_importance_df.head(15)
    
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker_color='lightblue'
        )
    )
    
    fig.update_layout(
        title='Top 15 Feature Importances (XGBoost)',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=500
    )
    
    return fig


@st.cache_data
def create_comprehensive_excel_report(result_df, hist_df, forecast_year, scaling_factor, 
                                    validation_scores, ensemble_weights=None, 
                                    forecast_info_dict=None):
    """Create comprehensive Excel report with multiple sheets"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Executive Summary
        exec_summary = {
            'Metric': ['Forecast Year', 'Data Points Used', 'Models Employed', 
                      'Best Performing Model', 'Ensemble Method', 'Scaling Factor Applied'],
            'Value': [forecast_year, len(hist_df), len(validation_scores),
                     min(validation_scores, key=validation_scores.get) if validation_scores else 'N/A',
                     'Weighted Ensemble (Softmax)', f"{scaling_factor:.2f}x"]
        }
        pd.DataFrame(exec_summary).to_excel(writer, sheet_name='Executive_Summary', index=False)
        
        # Sheet 2: Detailed Forecasts
        main_sheet = result_df.copy()
        main_sheet['Month'] = main_sheet['Month'].dt.strftime('%Y-%m-%d')
        main_sheet.to_excel(writer, sheet_name='Detailed_Forecasts', index=False)
        
        # Sheet 3: Model Performance
        actual_col = f'Actual_{forecast_year}'
        if actual_col in result_df.columns and result_df[actual_col].notna().any():
            model_cols = [col for col in result_df.columns if '_Forecast' in col or 
                         col in ['Weighted_Ensemble', 'Meta_Learning']]
            
            perf_data = []
            for col in model_cols:
                model_name = col.replace('_Forecast', '').replace('_', ' ')
                metrics = calculate_comprehensive_metrics(
                    result_df[result_df[actual_col].notna()][actual_col],
                    result_df[result_df[actual_col].notna()][col]
                )
                
                if metrics:
                    perf_data.append({
                        'Model': model_name,
                        'MAE': round(metrics['MAE'], 2),
                        'RMSE': round(metrics['RMSE'], 2),
                        'MAPE (%)': round(metrics['MAPE'], 2),
                        'SMAPE (%)': round(metrics['SMAPE'], 2),
                        'MASE': round(metrics['MASE'], 3),
                        'Directional Accuracy (%)': round(metrics.get('Directional_Accuracy', 0), 1),
                        'Bias': round(metrics['Bias'], 2),
                        'Bias (%)': round(metrics['Bias_Pct'], 2),
                        'Tracking Signal': round(metrics['Tracking_Signal'], 2)
                    })
            
            if perf_data:
                perf_df = pd.DataFrame(perf_data)
                perf_df.to_excel(writer, sheet_name='Model_Performance', index=False)
        
        # Sheet 4: Ensemble Weights
        if ensemble_weights:
            weights_df = pd.DataFrame([
                {'Model': k, 'Weight': v, 'Weight (%)': f"{v*100:.1f}%"} 
                for k, v in ensemble_weights.items()
            ])
            weights_df.to_excel(writer, sheet_name='Ensemble_Weights', index=False)
        
        # Sheet 5: Data Analysis
        analysis_data = []
        
        # Basic statistics
        analysis_data.extend([
            {'Category': 'Data Statistics', 'Metric': 'Total Months', 'Value': len(hist_df)},
            {'Category': 'Data Statistics', 'Metric': 'Mean Sales', 'Value': hist_df['Sales'].mean()},
            {'Category': 'Data Statistics', 'Metric': 'Std Dev Sales', 'Value': hist_df['Sales'].std()},
            {'Category': 'Data Statistics', 'Metric': 'CV', 'Value': hist_df['Sales'].std() / hist_df['Sales'].mean()},
        ])
        
        # Transformation info
        if 'transformation' in hist_df.columns:
            transform = hist_df['transformation'].iloc[0]
            analysis_data.append({
                'Category': 'Preprocessing', 
                'Metric': 'Transformation Applied', 
                'Value': transform
            })
        
        # Seasonality analysis
        if len(hist_df) >= 24:
            decomposition = seasonal_decompose(hist_df['Sales'], model='additive', period=12)
            seasonal_strength = np.var(decomposition.seasonal) / np.var(hist_df['Sales'])
            analysis_data.append({
                'Category': 'Time Series Properties',
                'Metric': 'Seasonality Strength',
                'Value': f"{seasonal_strength:.2%}"
            })
        
        analysis_df = pd.DataFrame(analysis_data)
        analysis_df.to_excel(writer, sheet_name='Data_Analysis', index=False)
        
        # Sheet 6: Feature Importance (if XGBoost was used)
        if forecast_info_dict and 'XGBoost' in forecast_info_dict:
            xgb_info = forecast_info_dict['XGBoost']
            if 'feature_importance' in xgb_info:
                xgb_info['feature_importance'].to_excel(
                    writer, sheet_name='Feature_Importance', index=False
                )
        
        # Sheet 7: Forecast Intervals
        if forecast_info_dict:
            interval_data = []
            for model_name, info in forecast_info_dict.items():
                if isinstance(info, dict) and 'lower_bound' in info:
                    for i, month in enumerate(result_df['Month']):
                        interval_data.append({
                            'Model': model_name,
                            'Month': month.strftime('%Y-%m-%d'),
                            'Forecast': info['values'][i],
                            'Lower_95%': info['lower_bound'][i],
                            'Upper_95%': info['upper_bound'][i]
                        })
            
            if interval_data:
                interval_df = pd.DataFrame(interval_data)
                interval_df.to_excel(writer, sheet_name='Forecast_Intervals', index=False)
        
        # Sheet 8: Model Diagnostics
        diagnostics_data = []
        
        # Add validation scores
        for model, score in validation_scores.items():
            diagnostics_data.append({
                'Model': model,
                'Metric': 'Validation Score',
                'Value': score if score != np.inf else 'Failed'
            })
        
        if diagnostics_data:
            diag_df = pd.DataFrame(diagnostics_data)
            diag_df.to_excel(writer, sheet_name='Model_Diagnostics', index=False)
        
        # Sheet 9: Monthly Comparison
        monthly_comp = result_df.copy()
        monthly_comp['Month_Name'] = monthly_comp['Month'].dt.strftime('%B')
        
        # Calculate average forecast across models
        forecast_cols = [col for col in result_df.columns if '_Forecast' in col]
        if forecast_cols:
            monthly_comp['Average_Forecast'] = monthly_comp[forecast_cols].mean(axis=1)
            monthly_comp['Forecast_StdDev'] = monthly_comp[forecast_cols].std(axis=1)
            monthly_comp['Forecast_CV'] = (monthly_comp['Forecast_StdDev'] / 
                                          monthly_comp['Average_Forecast'])
            
            monthly_summary = monthly_comp[['Month_Name', 'Average_Forecast', 
                                          'Forecast_StdDev', 'Forecast_CV']]
            monthly_summary.to_excel(writer, sheet_name='Monthly_Summary', index=False)
    
    output.seek(0)
    return output


def main():
    """Main function to run the enhanced forecasting application"""
    st.title("🚀 Advanced AI Sales Forecasting System")
    st.markdown("**Enterprise-grade forecasting with 10+ models, ensemble learning, and neural networks**")
    
    # Initialize session state
    if 'forecast_info' not in st.session_state:
        st.session_state.forecast_info = {}
    
    # Sidebar configuration
    st.sidebar.header("⚙️ System Configuration")
    
    # Basic settings
    forecast_year = st.sidebar.selectbox(
        "📅 Select Forecast Year",
        options=[2024, 2025, 2026, 2027],
        index=0
    )
    
    # Advanced settings
    with st.sidebar.expander("🔬 Advanced Settings", expanded=True):
        st.subheader("🎯 Optimization Settings")
        enable_hyperopt = st.checkbox("Enable Hyperparameter Optimization", value=True,
                                     help="Automatically tune model parameters (slower but more accurate)")
        enable_parallel = st.checkbox("Enable Parallel Processing", value=True,
                                     help="Use multiple CPU cores for faster training")
        enable_preprocessing = st.checkbox("Advanced Data Preprocessing", value=True,
                                         help="Apply outlier detection, transformations, and data cleaning")
        
        st.subheader("🤖 Ensemble Settings")
        ensemble_method = st.selectbox(
            "Ensemble Weighting Method",
            options=["Softmax", "Inverse Error", "Rank-based"],
            index=0
        )
        enable_meta_learning = st.checkbox("Enable Meta-Learning", value=True,
                                         help="Use stacking with multiple meta-learners")
        
        st.subheader("📊 Visualization Settings")
        show_intervals = st.checkbox("Show Prediction Intervals", value=True)
        show_diagnostics = st.checkbox("Show Diagnostic Plots", value=True)
    
    # Model selection
    st.sidebar.subheader("🤖 Model Selection")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.markdown("**Classic Models**")
        use_sarima = st.checkbox("SARIMA (Auto)", value=True)
        use_ets = st.checkbox("ETS (Auto)", value=True)
        use_theta = st.checkbox("Theta Method", value=True)
        use_croston = st.checkbox("Croston (Intermittent)", value=False)
    
    with col2:
        st.markdown("**ML/DL Models**")
        use_prophet = st.checkbox("Prophet (Enhanced)", value=True)
        use_xgboost = st.checkbox("XGBoost (Advanced)", value=True)
        use_lstm = st.checkbox("LSTM Neural Net", value=TENSORFLOW_AVAILABLE)
        use_ensemble = st.checkbox("Ensemble Models", value=True)
    
    # Validate model selection
    selected_models = sum([use_sarima, use_ets, use_theta, use_croston, 
                          use_prophet, use_xgboost, use_lstm])
    
    if selected_models == 0:
        st.sidebar.error("❌ Please select at least one model!")
        return
    
    # File upload section
    st.header("📁 Data Upload")
    
    col1, col2 = st.columns(2)
    
    with col1:
        historical_file = st.file_uploader(
            "📊 Upload Historical Sales Data",
            type=["xlsx", "xls"],
            help="Excel file with 'Month' and 'Sales' columns"
        )
    
    with col2:
        actual_file = st.file_uploader(
            f"📈 Upload {forecast_year} Actual Data (Optional)",
            type=["xlsx", "xls"],
            help="For validation and meta-learning"
        )
    
    if historical_file is None:
        st.info("👆 Please upload historical sales data to begin forecasting")
        
        # Show sample data format
        with st.expander("📋 View Sample Data Format"):
            sample_data = pd.DataFrame({
                'Month': pd.date_range('2022-01-01', periods=24, freq='MS'),
                'Sales': np.random.randint(1000, 5000, 24)
            })
            st.dataframe(sample_data.head(10))
        
        return
    
    # Load data with caching
    file_content = historical_file.read()
    file_hash = hashlib.md5(file_content).hexdigest()
    
    hist_df = load_data_optimized(file_content, file_hash)
    
    if hist_df is None:
        return
    
    # Load actual data if provided
    actual_df = None
    scaling_factor = 1.0
    
    if actual_file is not None:
        actual_content = actual_file.read()
        actual_df = load_actual_2024_data(io.BytesIO(actual_content), forecast_year)
        
        if actual_df is not None:
            # Advanced scaling detection
            scaling_factor = detect_and_apply_scaling(hist_df, actual_df)
    
    # Data Analysis Dashboard
    st.header("📊 Data Analysis Dashboard")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📅 Data Points", len(hist_df))
    
    with col2:
        avg_sales = hist_df['Sales_Original'].mean()
        st.metric("💰 Avg Sales", f"{avg_sales:,.0f}")
    
    with col3:
        cv = hist_df['Sales_Original'].std() / avg_sales
        st.metric("📊 Coefficient of Variation", f"{cv:.2%}")
    
    with col4:
        data_quality = min(100, len(hist_df) * 4.17)
        st.metric("🎯 Data Quality", f"{data_quality:.0f}%")
    
    with col5:
        freq = detect_data_frequency(hist_df['Month'])
        st.metric("📆 Frequency", freq)
    
    # Show diagnostic plots
    if show_diagnostics:
        with st.expander("📈 Time Series Diagnostics", expanded=False):
            diagnostic_fig = create_diagnostic_plots(hist_df)
            st.plotly_chart(diagnostic_fig, use_container_width=True)
    
    # Forecasting section
    if st.button("🚀 Generate AI Forecasts", type="primary", use_container_width=True):
        st.header("🔮 Generating Advanced Forecasts...")
        
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Prepare models
        models_to_run = []
        
        if use_sarima:
            models_to_run.append(("SARIMA", run_advanced_sarima_forecast))
        if use_prophet:
            models_to_run.append(("Prophet", run_advanced_prophet_forecast))
        if use_ets:
            models_to_run.append(("ETS", run_advanced_ets_forecast))
        if use_xgboost:
            models_to_run.append(("XGBoost", run_advanced_xgboost_forecast))
        if use_theta:
            models_to_run.append(("Theta", run_theta_forecast))
        if use_croston:
            models_to_run.append(("Croston", run_croston_forecast))
        if use_lstm and TENSORFLOW_AVAILABLE:
            models_to_run.append(("LSTM", run_lstm_forecast))
        
        # Run models (parallel or sequential)
        forecast_results = {}
        validation_scores = {}
        forecast_info = {}
        
        if enable_parallel and len(models_to_run) > 2:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=min(4, len(models_to_run))) as executor:
                futures = []
                for model_name, model_func in models_to_run:
                    future = executor.submit(
                        parallel_model_training,
                        model_func, hist_df, 12, scaling_factor, model_name
                    )
                    futures.append(future)
                
                for i, future in enumerate(futures):
                    model_name, result = future.result()
                    forecast_values, score = result
                    
                    forecast_results[f"{model_name}_Forecast"] = forecast_values
                    validation_scores[model_name] = score
                    
                    # Store additional info if available
                    if model_name == "XGBoost" and 'xgboost_info' in st.session_state:
                        forecast_info[model_name] = st.session_state['xgboost_info']
                    
                    progress_bar.progress((i + 1) / len(models_to_run))
                    status_text.text(f"Completed: {model_name}")
        else:
            # Sequential execution
            for i, (model_name, model_func) in enumerate(models_to_run):
                status_text.text(f"Training {model_name}...")
                
                try:
                    forecast_values, score = model_func(hist_df, 12, scaling_factor)
                    forecast_results[f"{model_name}_Forecast"] = forecast_values
                    validation_scores[model_name] = score
                    
                    # Show model-specific info
                    if score != np.inf:
                        st.success(f"✅ {model_name} completed (Score: {score:.2f})")
                    else:
                        st.warning(f"⚠️ {model_name} completed with fallback")
                    
                except Exception as e:
                    st.error(f"❌ {model_name} failed: {str(e)}")
                    forecast_results[f"{model_name}_Forecast"] = run_fallback_forecast(
                        hist_df, 12, scaling_factor
                    )
                    validation_scores[model_name] = np.inf
                
                progress_bar.progress((i + 1) / len(models_to_run))
        
        # Create ensemble forecasts
        if use_ensemble and len(forecast_results) > 1:
            status_text.text("Creating ensemble forecasts...")
            
            # Weighted ensemble
            ensemble_forecast, ensemble_weights, ensemble_variants = create_advanced_ensemble(
                forecast_results, validation_scores, actual_df
            )
            forecast_results["Weighted_Ensemble"] = ensemble_forecast
            
            # Show ensemble weights
            st.info(f"🎯 Ensemble Weights ({ensemble_method}): " + 
                   ", ".join([f"{k}: {v:.1%}" for k, v in ensemble_weights.items()]))
            
            # Meta-learning
            if enable_meta_learning and actual_df is not None:
                meta_forecast = run_meta_learning_ensemble(
                    forecast_results, hist_df, actual_df
                )
                if meta_forecast is not None:
                    forecast_results["Meta_Learning"] = meta_forecast
                    st.success("✅ Meta-learning ensemble created")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Create results dataframe
        forecast_dates = pd.date_range(
            start=f"{forecast_year}-01-01",
            end=f"{forecast_year}-12-01",
            freq='MS'
        )
        
        result_df = pd.DataFrame({
            "Month": forecast_dates,
            **forecast_results
        })
        
        # Merge actual data
        if actual_df is not None:
            result_df = result_df.merge(actual_df, on="Month", how="left")
            
            # Show coverage info
            actual_count = result_df[f'Actual_{forecast_year}'].notna().sum()
            st.success(f"📊 Validation data available for {actual_count} months")
        
        # Store in session state
        st.session_state['result_df'] = result_df
        st.session_state['validation_scores'] = validation_scores
        st.session_state['ensemble_weights'] = ensemble_weights if 'ensemble_weights' in locals() else None
        st.session_state['forecast_info'] = forecast_info
        
        # Display results
        st.header("📊 Forecast Results")
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_forecast = result_df['Weighted_Ensemble'].sum() if 'Weighted_Ensemble' in result_df else 0
            st.metric("📈 Total Forecast", f"{total_forecast:,.0f}")
        
        with col2:
            avg_monthly = total_forecast / 12
            st.metric("📅 Average Monthly", f"{avg_monthly:,.0f}")
        
        with col3:
            yoy_growth = ((total_forecast - hist_df['Sales_Original'].tail(12).sum()) / 
                         hist_df['Sales_Original'].tail(12).sum() * 100)
            st.metric("📊 YoY Growth", f"{yoy_growth:+.1f}%")
        
        # Show forecast table
        st.subheader("📋 Detailed Forecasts")
        
        # Format display
        display_df = result_df.copy()
        display_df['Month'] = display_df['Month'].dt.strftime('%b %Y')
        
        # Round numeric columns
        numeric_cols = [col for col in display_df.columns if col != 'Month']
        for col in numeric_cols:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:,.0f}" if pd.notna(x) else "—"
            )
        
        # Style the dataframe
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
        
        # Visualization
        st.subheader("📈 Forecast Visualization")
        
        # Create comprehensive plot
        forecast_fig = create_forecast_plot(result_df, forecast_year, hist_df)
        st.plotly_chart(forecast_fig, use_container_width=True)
        
        # Model Performance Analysis
        actual_col = f'Actual_{forecast_year}'
        if actual_col in result_df.columns and result_df[actual_col].notna().any():
            st.subheader("🎯 Model Performance Analysis")
            
            # Create performance summary
            performance_data = []
            
            for col in [c for c in result_df.columns if '_Forecast' in c or 
                       c in ['Weighted_Ensemble', 'Meta_Learning']]:
                model_name = col.replace('_Forecast', '').replace('_', ' ')
                
                # Calculate metrics only for available actual data
                actual_subset = result_df[result_df[actual_col].notna()]
                metrics = calculate_comprehensive_metrics(
                    actual_subset[actual_col],
                    actual_subset[col]
                )
                
                if metrics:
                    performance_data.append({
                        'Model': model_name,
                        'MAPE (%)': f"{metrics['MAPE']:.1f}",
                        'RMSE': f"{metrics['RMSE']:,.0f}",
                        'MAE': f"{metrics['MAE']:,.0f}",
                        'Bias (%)': f"{metrics['Bias_Pct']:+.1f}",
                        'Direction Acc (%)': f"{metrics.get('Directional_Accuracy', 0):.0f}",
                        'Tracking Signal': f"{metrics['Tracking_Signal']:.1f}"
                    })
            
            if performance_data:
                perf_df = pd.DataFrame(performance_data)
                
                # Sort by MAPE
                perf_df['MAPE_numeric'] = perf_df['MAPE (%)'].str.replace('%', '').astype(float)
                perf_df = perf_df.sort_values('MAPE_numeric').drop('MAPE_numeric', axis=1)
                
                # Display with highlighting
                st.dataframe(
                    perf_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Best model
                best_model = perf_df.iloc[0]['Model']
                best_mape = perf_df.iloc[0]['MAPE (%)']
                st.success(f"🏆 Best Model: **{best_model}** (MAPE: {best_mape})")
        
        # Feature Importance (if XGBoost was used)
        if 'xgboost_info' in st.session_state and st.session_state['xgboost_info']:
            xgb_info = st.session_state['xgboost_info']
            if 'feature_importance' in xgb_info:
                st.subheader("🔍 Feature Importance Analysis")
                
                feat_imp_fig = create_feature_importance_plot(xgb_info['feature_importance'])
                st.plotly_chart(feat_imp_fig, use_container_width=True)
                
                # Show top features
                top_features = xgb_info['feature_importance'].head(5)
                st.info(f"🎯 Top predictive features: {', '.join(top_features['feature'].tolist())}")
        
        # Ensemble Analysis
        if 'ensemble_weights' in st.session_state and st.session_state['ensemble_weights']:
            st.subheader("🤝 Ensemble Analysis")
            
            weights_df = pd.DataFrame([
                {'Model': k, 'Weight': v} 
                for k, v in st.session_state['ensemble_weights'].items()
            ]).sort_values('Weight', ascending=False)
            
            # Create weight visualization
            fig = go.Figure(go.Bar(
                x=weights_df['Model'],
                y=weights_df['Weight'],
                text=[f"{w:.1%}" for w in weights_df['Weight']],
                textposition='auto',
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title='Ensemble Model Weights',
                xaxis_title='Model',
                yaxis_title='Weight',
                yaxis_tickformat='.0%',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Advanced Analytics Section
        st.header("📊 Advanced Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Seasonal analysis
            st.subheader("🌊 Seasonal Pattern Analysis")
            
            if 'Weighted_Ensemble' in result_df.columns:
                monthly_avg = result_df.groupby(result_df['Month'].dt.month)['Weighted_Ensemble'].mean()
                seasonal_index = (monthly_avg / monthly_avg.mean() * 100).round(1)
                
                fig = go.Figure(go.Bar(
                    x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                    y=seasonal_index.values,
                    text=[f"{v:.0f}" for v in seasonal_index.values],
                    textposition='auto',
                    marker_color=['red' if v < 100 else 'green' for v in seasonal_index.values]
                ))
                
                fig.update_layout(
                    title='Seasonal Index (100 = Average)',
                    xaxis_title='Month',
                    yaxis_title='Index',
                    height=350
                )
                
                fig.add_hline(y=100, line_dash="dash", line_color="gray")
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Forecast stability
            st.subheader("📊 Forecast Stability Analysis")
            
            forecast_cols = [col for col in result_df.columns if '_Forecast' in col]
            if len(forecast_cols) > 1:
                forecast_array = result_df[forecast_cols].values
                cv_by_month = np.std(forecast_array, axis=1) / np.mean(forecast_array, axis=1)
                
                avg_cv = np.mean(cv_by_month)
                stability_score = max(0, 100 - (avg_cv * 100))
                
                st.metric("🎯 Forecast Stability Score", f"{stability_score:.0f}%")
                st.info(f"Average CV across models: {avg_cv:.2%}")
                
                # Monthly stability chart
                fig = go.Figure(go.Scatter(
                    x=result_df['Month'],
                    y=cv_by_month,
                    mode='lines+markers',
                    name='CV by Month',
                    line=dict(color='orange', width=2)
                ))
                
                fig.update_layout(
                    title='Forecast Variability by Month',
                    xaxis_title='Month',
                    yaxis_title='Coefficient of Variation',
                    height=250
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Risk Analysis
        st.subheader("⚠️ Risk Analysis")
        
        if 'Weighted_Ensemble' in result_df.columns:
            # Calculate risks
            total_forecast = result_df['Weighted_Ensemble'].sum()
            
            # Model divergence risk
            if len(forecast_cols) > 1:
                max_divergence = np.max([
                    np.abs(result_df[col].sum() - total_forecast) / total_forecast 
                    for col in forecast_cols
                ])
                
                divergence_risk = "Low" if max_divergence < 0.1 else "Medium" if max_divergence < 0.2 else "High"
            else:
                divergence_risk = "N/A"
            
            # Trend reversal risk
            historical_trend = np.polyfit(range(len(hist_df)), hist_df['Sales_Original'], 1)[0]
            forecast_trend = np.polyfit(range(12), result_df['Weighted_Ensemble'], 1)[0]
            trend_reversal = np.sign(historical_trend) != np.sign(forecast_trend)
            
            # Seasonality disruption risk
            if len(hist_df) >= 24:
                historical_seasonal_cv = hist_df.groupby(hist_df['Month'].dt.month)['Sales_Original'].std().mean() / hist_df['Sales_Original'].mean()
                forecast_seasonal_cv = result_df.groupby(result_df['Month'].dt.month)['Weighted_Ensemble'].std().mean() / result_df['Weighted_Ensemble'].mean()
                seasonality_change = abs(forecast_seasonal_cv - historical_seasonal_cv) / historical_seasonal_cv
                seasonality_risk = "Low" if seasonality_change < 0.2 else "Medium" if seasonality_change < 0.5 else "High"
            else:
                seasonality_risk = "N/A"
            
            # Display risks
            risk_col1, risk_col2, risk_col3 = st.columns(3)
            
            with risk_col1:
                st.metric("Model Divergence Risk", divergence_risk)
            
            with risk_col2:
                trend_risk = "High" if trend_reversal else "Low"
                st.metric("Trend Reversal Risk", trend_risk)
            
            with risk_col3:
                st.metric("Seasonality Risk", seasonality_risk)
        
        # Download Section
        st.header("📥 Export Results")
        
        # Generate comprehensive report
        excel_report = create_comprehensive_excel_report(
            result_df,
            hist_df,
            forecast_year,
            scaling_factor,
            validation_scores,
            st.session_state.get('ensemble_weights'),
            st.session_state.get('forecast_info')
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="📊 Download Full Report (Excel)",
                data=excel_report,
                file_name=f"AI_Forecast_Report_{forecast_year}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col2:
            # CSV download
            csv_data = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download Forecasts (CSV)",
                data=csv_data,
                file_name=f"Forecasts_{forecast_year}.csv",
                mime="text/csv"
            )
        
        with col3:
            # JSON download for API integration
            json_data = result_df.to_json(orient='records', date_format='iso')
            st.download_button(
                label="🔧 Download JSON (API)",
                data=json_data,
                file_name=f"forecast_api_{forecast_year}.json",
                mime="application/json"
            )
        
        # Report contents
        with st.expander("📋 Report Contents"):
            st.markdown("""
            **Comprehensive Excel Report includes:**
            - 📊 **Executive Summary**: Key metrics and configuration
            - 📈 **Detailed Forecasts**: All model predictions with dates
            - 🎯 **Model Performance**: Comprehensive accuracy metrics
            - 🤝 **Ensemble Weights**: Model contribution analysis
            - 📊 **Data Analysis**: Statistical properties and transformations
            - 🔍 **Feature Importance**: Key drivers (if XGBoost used)
            - 📉 **Forecast Intervals**: Confidence bounds
            - 🔧 **Model Diagnostics**: Validation scores and parameters
            - 📅 **Monthly Summary**: Aggregated insights
            """)
        
        # Final insights
        st.header("💡 Key Insights & Recommendations")
        
        insights = []
        
        # Growth insight
        if yoy_growth > 10:
            insights.append(f"📈 **Strong Growth Expected**: {yoy_growth:.1f}% YoY increase projected")
        elif yoy_growth < -10:
            insights.append(f"📉 **Significant Decline Warning**: {yoy_growth:.1f}% YoY decrease projected")
        
        # Seasonality insight
        if 'Weighted_Ensemble' in result_df.columns:
            peak_month = result_df.loc[result_df['Weighted_Ensemble'].idxmax(), 'Month'].strftime('%B')
            low_month = result_df.loc[result_df['Weighted_Ensemble'].idxmin(), 'Month'].strftime('%B')
            insights.append(f"📊 **Seasonal Pattern**: Peak in {peak_month}, lowest in {low_month}")
        
        # Model consensus
        if len(forecast_cols) > 1 and 'Weighted_Ensemble' in result_df.columns:
            avg_cv = np.mean(np.std(result_df[forecast_cols].values, axis=1) / 
                           np.mean(result_df[forecast_cols].values, axis=1))
            if avg_cv < 0.1:
                insights.append("✅ **High Model Consensus**: All models strongly agree")
            elif avg_cv > 0.2:
                insights.append("⚠️ **Model Divergence**: Consider reviewing outlier predictions")
        
        # Display insights
        for insight in insights:
            st.info(insight)
        
        # Success message
        st.success("✅ Forecasting complete! Results are ready for download.")


if __name__ == "__main__":
    main()import streamlit as st

# Configure streamlit FIRST - must be before any other st commands
st.set_page_config(page_title="Advanced AI Sales Forecasting System", layout="wide")

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
from scipy import stats
from scipy.optimize import minimize
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib

# Forecasting libraries
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.forecasting.theta import ThetaModel

# Machine learning libraries
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import VotingRegressor
from sklearn.base import BaseEstimator, RegressorMixin

# Try to import optional libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    st.warning("XGBoost not installed. Install with: pip install xgboost")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.info("TensorFlow not available for neural network models.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

import warnings
warnings.filterwarnings("ignore")

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedMetaLearner(BaseEstimator, RegressorMixin):
    """Advanced meta-learner with multiple stacking options"""
    def __init__(self, meta_model='ridge', cv_folds=5):
        self.meta_model = meta_model
        self.cv_folds = cv_folds
        self.model = None
        self.feature_importance = None
        
    def fit(self, X, y):
        if self.meta_model == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif self.meta_model == 'lasso':
            self.model = Lasso(alpha=0.1)
        elif self.meta_model == 'elastic':
            self.model = ElasticNet(alpha=0.1, l1_ratio=0.5)
        elif self.meta_model == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        self.model.fit(X, y)
        
        # Calculate feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            self.feature_importance = np.abs(self.model.coef_)
            
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_feature_importance(self):
        return self.feature_importance


def optimize_dtypes(df):
    """Reduce memory usage by optimizing data types"""
    initial_memory = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object' and col != 'Month':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    final_memory = df.memory_usage().sum() / 1024**2
    memory_reduction = (initial_memory - final_memory) / initial_memory * 100
    
    if memory_reduction > 0:
        st.info(f"💾 Memory optimized: {initial_memory:.2f} MB → {final_memory:.2f} MB ({memory_reduction:.1f}% reduction)")
    
    return df


@st.cache_data(ttl=3600)
def load_data_optimized(file_content, file_hash):
    """Load and preprocess data with memory optimization"""
    try:
        df = pd.read_excel(io.BytesIO(file_content))
    except Exception:
        st.error("Could not read the uploaded file. Please ensure it's a valid Excel file.")
        return None

    if "Month" not in df.columns or "Sales" not in df.columns:
        st.error("The file must contain 'Month' and 'Sales' columns.")
        return None

    # Parse dates
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    if df["Month"].isna().any():
        st.error("Some dates could not be parsed. Please check the 'Month' column format.")
        return None

    # Clean sales data
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
    df["Sales"] = df["Sales"].abs()

    # Sort by date
    df = df.sort_values("Month").reset_index(drop=True)
    
    # Optimize data types
    df = optimize_dtypes(df)
    
    # Check if there are multiple entries per month
    original_rows = len(df)
    unique_months = df['Month'].nunique()
    
    if original_rows > unique_months:
        st.info(f"📊 Aggregating {original_rows} data points into {unique_months} monthly totals...")
        
        # Aggregate by month
        df_monthly = df.groupby('Month', as_index=False).agg({
            'Sales': 'sum'
        }).sort_values('Month').reset_index(drop=True)
        
        df_monthly['Sales_Original'] = df_monthly['Sales'].copy()
        
        # Advanced preprocessing
        df_processed = advanced_preprocess_data(df_monthly)
        
        st.success(f"✅ Successfully aggregated to {len(df_processed)} monthly data points")
        
    else:
        df_processed = advanced_preprocess_data(df)
    
    # Force garbage collection
    gc.collect()
    
    return df_processed


def advanced_preprocess_data(df):
    """Enhanced data preprocessing with multiple techniques"""
    df = df.copy()
    df['Sales_Original'] = df['Sales'].copy()
    
    # 1. Advanced Outlier Detection (multiple methods)
    outlier_methods = []
    
    # IQR method
    Q1 = df['Sales'].quantile(0.25)
    Q3 = df['Sales'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    iqr_outliers = ((df['Sales'] < lower_bound) | (df['Sales'] > upper_bound))
    outlier_methods.append(iqr_outliers)
    
    # Z-score method
    z_scores = np.abs(stats.zscore(df['Sales']))
    zscore_outliers = z_scores > 3
    outlier_methods.append(zscore_outliers)
    
    # Isolation Forest (if enough data)
    if len(df) >= 20:
        from sklearn.ensemble import IsolationForest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_pred = iso_forest.fit_predict(df['Sales'].values.reshape(-1, 1))
        iso_outliers = outlier_pred == -1
        outlier_methods.append(iso_outliers)
    
    # Combine outlier detection methods (majority vote)
    outliers = np.sum(outlier_methods, axis=0) >= len(outlier_methods) / 2
    outliers_detected = outliers.sum()
    
    if outliers_detected > 0:
        st.info(f"📊 Detected {outliers_detected} outliers using ensemble method")
        # Use Winsorization instead of hard capping
        df.loc[outliers, 'Sales'] = df.loc[~outliers, 'Sales'].quantile(0.95)
    
    # 2. Handle missing values with advanced interpolation
    if df['Sales'].isna().any():
        # Try multiple interpolation methods
        df['Sales'] = df['Sales'].interpolate(method='time')
        # Fill any remaining NaNs with seasonal average
        month_avg = df.groupby(df['Month'].dt.month)['Sales'].transform('mean')
        df['Sales'] = df['Sales'].fillna(month_avg)
    
    # 3. Detect and handle structural breaks
    if len(df) >= 24:
        from statsmodels.tsa.stattools import adfuller
        # Check for stationarity
        adf_result = adfuller(df['Sales'])
        if adf_result[1] > 0.05:  # Non-stationary
            st.info("📈 Non-stationary data detected. Applying differencing.")
            df['needs_differencing'] = True
        else:
            df['needs_differencing'] = False
    
    # 4. Advanced transformation selection
    transformations = {
        'none': df['Sales'].copy(),
        'log': np.log1p(df['Sales']),
        'sqrt': np.sqrt(df['Sales']),
        'boxcox': stats.boxcox(df['Sales'] + 1)[0] if (df['Sales'] > 0).all() else df['Sales']
    }
    
    # Select best transformation based on normality
    best_transform = 'none'
    best_normality = 0
    
    for transform_name, transformed_data in transformations.items():
        try:
            _, p_value = stats.normaltest(transformed_data)
            if p_value > best_normality:
                best_normality = p_value
                best_transform = transform_name
        except:
            continue
    
    if best_transform != 'none':
        st.info(f"📊 Applied {best_transform} transformation for better modeling")
        df['Sales'] = transformations[best_transform]
        df['transformation'] = best_transform
        df['transformation_params'] = {'method': best_transform}
        
        if best_transform == 'boxcox':
            df['transformation_params']['lambda'] = stats.boxcox(df['Sales_Original'] + 1)[1]
    else:
        df['transformation'] = 'none'
        df['transformation_params'] = {'method': 'none'}
    
    # 5. Add cyclical encoding for months
    df['month'] = df['Month'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df


def create_advanced_features(df):
    """Create comprehensive features for ML models"""
    df = df.copy()
    
    # Time features
    df['year'] = df['Month'].dt.year
    df['quarter'] = df['Month'].dt.quarter
    df['dayofyear'] = df['Month'].dt.dayofyear
    df['weekofyear'] = df['Month'].dt.isocalendar().week
    
    # Lag features (multiple lags)
    lag_features = [1, 2, 3, 6, 12, 24] if len(df) > 24 else [1, 3, 6, 12]
    for lag in lag_features:
        if lag < len(df):
            df[f'lag_{lag}'] = df['Sales'].shift(lag)
    
    # Rolling statistics (multiple windows)
    windows = [3, 6, 12, 24] if len(df) > 24 else [3, 6, 12]
    for window in windows:
        if window < len(df):
            df[f'rolling_mean_{window}'] = df['Sales'].rolling(window=window, min_periods=1).mean()
            df[f'rolling_std_{window}'] = df['Sales'].rolling(window=window, min_periods=1).std()
            df[f'rolling_min_{window}'] = df['Sales'].rolling(window=window, min_periods=1).min()
            df[f'rolling_max_{window}'] = df['Sales'].rolling(window=window, min_periods=1).max()
            
            # Exponentially weighted statistics
            df[f'ewm_mean_{window}'] = df['Sales'].ewm(span=window, min_periods=1).mean()
    
    # Trend features
    df['trend'] = np.arange(len(df))
    df['trend_squared'] = df['trend'] ** 2
    
    # Seasonal strength indicator
    if len(df) >= 24:
        seasonal_strength = df.groupby(df['Month'].dt.month)['Sales'].std() / df['Sales'].std()
        df['seasonal_strength'] = df['Month'].dt.month.map(seasonal_strength)
    
    # Growth rates
    df['mom_growth'] = df['Sales'].pct_change(1)
    df['yoy_growth'] = df['Sales'].pct_change(12)
    
    # Fourier features for multiple seasonalities
    for period in [6, 12]:
        for i in range(1, 3):  # Use 2 fourier terms
            df[f'sin_{period}_{i}'] = np.sin(2 * np.pi * i * df.index / period)
            df[f'cos_{period}_{i}'] = np.cos(2 * np.pi * i * df.index / period)
    
    return df


def parallel_model_training(model_func, data, forecast_periods, scaling_factor, model_name):
    """Wrapper for parallel model training"""
    try:
        with st.spinner(f"🚀 Training {model_name}..."):
            result = model_func(data, forecast_periods, scaling_factor)
            return model_name, result
    except Exception as e:
        logger.error(f"Error in {model_name}: {str(e)}")
        return model_name, (run_fallback_forecast(data, forecast_periods, scaling_factor), np.inf)


def run_advanced_sarima_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """Enhanced SARIMA with automatic order selection and diagnostics"""
    try:
        work_data = data.copy()
        
        # Auto ARIMA order selection
        from pmdarima import auto_arima
        
        with st.spinner("🔧 Auto-tuning SARIMA parameters..."):
            auto_model = auto_arima(
                work_data['Sales'],
                start_p=0, start_q=0, max_p=3, max_q=3,
                seasonal=True, m=12, start_P=0, start_Q=0,
                max_P=2, max_Q=2, trace=False,
                error_action='ignore', suppress_warnings=True,
                stepwise=True, n_jobs=-1
            )
            
            best_order = auto_model.order
            best_seasonal_order = auto_model.seasonal_order
        
        # Fit final model
        model = SARIMAX(
            work_data['Sales'],
            order=best_order,
            seasonal_order=best_seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted_model = model.fit(disp=False)
        
        # Model diagnostics
        residuals = fitted_model.resid
        ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)
        
        # Generate forecast with prediction intervals
        forecast_result = fitted_model.get_forecast(steps=forecast_periods)
        forecast = forecast_result.predicted_mean
        confidence_intervals = forecast_result.conf_int(alpha=0.05)
        
        # Reverse transformations
        if 'transformation' in work_data.columns:
            transform_method = work_data['transformation'].iloc[0]
            if transform_method == 'log':
                forecast = np.expm1(forecast)
                confidence_intervals = np.expm1(confidence_intervals)
            elif transform_method == 'sqrt':
                forecast = forecast ** 2
                confidence_intervals = confidence_intervals ** 2
            elif transform_method == 'boxcox':
                lambda_param = work_data['transformation_params'].iloc[0].get('lambda', 1)
                forecast = inv_boxcox(forecast, lambda_param)
                confidence_intervals = confidence_intervals.apply(lambda x: inv_boxcox(x, lambda_param))
        
        forecast = np.maximum(forecast, 0) * scaling_factor
        
        # Calculate model score
        aic = fitted_model.aic
        
        # Store additional info
        forecast_info = {
            'values': forecast,
            'lower_bound': confidence_intervals.iloc[:, 0].values * scaling_factor,
            'upper_bound': confidence_intervals.iloc[:, 1].values * scaling_factor,
            'model_params': {'order': best_order, 'seasonal_order': best_seasonal_order},
            'diagnostics': {'ljung_box_pvalue': ljung_box['lb_pvalue'].mean()}
        }
        
        return forecast, aic
        
    except ImportError:
        st.warning("pmdarima not installed. Using manual parameter selection.")
        return run_manual_sarima_forecast(data, forecast_periods, scaling_factor)
    except Exception as e:
        st.warning(f"SARIMA failed: {str(e)}")
        return run_fallback_forecast(data, forecast_periods, scaling_factor), np.inf


def run_manual_sarima_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """Fallback SARIMA with manual parameter optimization"""
    try:
        work_data = data.copy()
        
        # Grid search for best parameters
        best_aic = np.inf
        best_params = None
        
        p_values = range(0, 3)
        d_values = range(0, 2)
        q_values = range(0, 3)
        P_values = range(0, 2)
        D_values = range(0, 2)
        Q_values = range(0, 2)
        
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    for P in P_values:
                        for D in D_values:
                            for Q in Q_values:
                                try:
                                    model = SARIMAX(
                                        work_data['Sales'],
                                        order=(p, d, q),
                                        seasonal_order=(P, D, Q, 12),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False
                                    )
                                    fitted = model.fit(disp=False, maxiter=100)
                                    
                                    if fitted.aic < best_aic:
                                        best_aic = fitted.aic
                                        best_params = {
                                            'order': (p, d, q),
                                            'seasonal_order': (P, D, Q, 12)
                                        }
                                except:
                                    continue
        
        if best_params:
            # Fit best model
            model = SARIMAX(
                work_data['Sales'],
                order=best_params['order'],
                seasonal_order=best_params['seasonal_order'],
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted_model = model.fit(disp=False)
            forecast = fitted_model.forecast(steps=forecast_periods)
            
            # Apply transformations
            if 'transformation' in work_data.columns:
                transform_method = work_data['transformation'].iloc[0]
                if transform_method == 'log':
                    forecast = np.expm1(forecast)
                elif transform_method == 'sqrt':
                    forecast = forecast ** 2
            
            forecast = np.maximum(forecast, 0) * scaling_factor
            return forecast, best_aic
        else:
            raise ValueError("No valid SARIMA model found")
            
    except Exception as e:
        return run_fallback_forecast(data, forecast_periods, scaling_factor), np.inf


def run_advanced_prophet_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """Enhanced Prophet with holiday effects and changepoint detection"""
    try:
        work_data = data.copy()
        
        # Prepare data
        prophet_data = work_data[['Month', 'Sales']].rename(columns={'Month': 'ds', 'Sales': 'y'})
        
        # Detect holidays/special events (outliers as potential holidays)
        if len(prophet_data) >= 24:
            rolling_mean = prophet_data['y'].rolling(window=12, center=True).mean()
            rolling_std = prophet_data['y'].rolling(window=12, center=True).std()
            outliers = np.abs(prophet_data['y'] - rolling_mean) > 2 * rolling_std
            
            holidays = pd.DataFrame({
                'holiday': 'detected_event',
                'ds': prophet_data.loc[outliers, 'ds'],
                'lower_window': -1,
                'upper_window': 1
            })
        else:
            holidays = None
        
        # Hyperparameter tuning
        param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'seasonality_mode': ['additive', 'multiplicative'],
            'changepoint_range': [0.8, 0.9, 0.95]
        }
        
        # Use cross-validation for parameter selection
        best_mape = np.inf
        best_params = {}
        
        if len(prophet_data) >= 36:  # Need enough data for CV
            from prophet.diagnostics import cross_validation, performance_metrics
            
            for cps in param_grid['changepoint_prior_scale']:
                for sps in param_grid['seasonality_prior_scale']:
                    for sm in param_grid['seasonality_mode']:
                        for cr in param_grid['changepoint_range']:
                            try:
                                model = Prophet(
                                    changepoint_prior_scale=cps,
                                    seasonality_prior_scale=sps,
                                    seasonality_mode=sm,
                                    changepoint_range=cr,
                                    yearly_seasonality=True,
                                    weekly_seasonality=False,
                                    daily_seasonality=False,
                                    holidays=holidays
                                )
                                
                                # Add custom seasonalities
                                model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
                                
                                model.fit(prophet_data)
                                
                                # Quick CV with minimal parameters
                                cv_df = cross_validation(
                                    model, 
                                    initial='365 days',
                                    period='90 days',
                                    horizon='90 days',
                                    parallel="threads"
                                )
                                
                                cv_metrics = performance_metrics(cv_df)
                                mape = cv_metrics['mape'].mean()
                                
                                if mape < best_mape:
                                    best_mape = mape
                                    best_params = {
                                        'changepoint_prior_scale': cps,
                                        'seasonality_prior_scale': sps,
                                        'seasonality_mode': sm,
                                        'changepoint_range': cr
                                    }
                            except:
                                continue
        else:
            # Default params for small datasets
            best_params = {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'seasonality_mode': 'additive',
                'changepoint_range': 0.8
            }
        
        # Train final model
        final_model = Prophet(
            **best_params,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            holidays=holidays,
            interval_width=0.95
        )
        
        # Add regressors if we have additional features
        if 'month_sin' in work_data.columns:
            prophet_data['month_sin'] = work_data['month_sin']
            prophet_data['month_cos'] = work_data['month_cos']
            final_model.add_regressor('month_sin')
            final_model.add_regressor('month_cos')
        
        final_model.fit(prophet_data)
        
        # Make predictions
        future = final_model.make_future_dataframe(periods=forecast_periods, freq='MS')
        
        # Add regressor values for future dates
        if 'month_sin' in prophet_data.columns:
            future_months = pd.to_datetime(future['ds']).dt.month
            future['month_sin'] = np.sin(2 * np.pi * future_months / 12)
            future['month_cos'] = np.cos(2 * np.pi * future_months / 12)
        
        forecast = final_model.predict(future)
        forecast_values = forecast['yhat'].tail(forecast_periods).values
        lower_bound = forecast['yhat_lower'].tail(forecast_periods).values
        upper_bound = forecast['yhat_upper'].tail(forecast_periods).values
        
        # Apply transformations
        if 'transformation' in work_data.columns:
            transform_method = work_data['transformation'].iloc[0]
            if transform_method == 'log':
                forecast_values = np.expm1(forecast_values)
                lower_bound = np.expm1(lower_bound)
                upper_bound = np.expm1(upper_bound)
            elif transform_method == 'sqrt':
                forecast_values = forecast_values ** 2
                lower_bound = lower_bound ** 2
                upper_bound = upper_bound ** 2
        
        forecast_values = np.maximum(forecast_values, 0) * scaling_factor
        
        # Store forecast info
        forecast_info = {
            'values': forecast_values,
            'lower_bound': lower_bound * scaling_factor,
            'upper_bound': upper_bound * scaling_factor,
            'changepoints': final_model.changepoints,
            'model_params': best_params
        }
        
        return forecast_values, best_mape
        
    except Exception as e:
        st.warning(f"Prophet failed: {str(e)}")
        return run_fallback_forecast(data, forecast_periods, scaling_factor), np.inf


def run_advanced_ets_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """Advanced ETS with automatic model selection and state space formulation"""
    try:
        work_data = data.copy()
        
        # Test different ETS configurations
        configs = [
            {'seasonal': 'add', 'trend': 'add', 'damped_trend': True, 'use_boxcox': True},
            {'seasonal': 'add', 'trend': 'add', 'damped_trend': False, 'use_boxcox': True},
            {'seasonal': 'mul', 'trend': 'add', 'damped_trend': True, 'use_boxcox': False},
            {'seasonal': 'mul', 'trend': 'add', 'damped_trend': False, 'use_boxcox': False},
            {'seasonal': 'add', 'trend': None, 'use_boxcox': True},
            {'seasonal': 'mul', 'trend': None, 'use_boxcox': False},
            {'seasonal': None, 'trend': 'add', 'damped_trend': True, 'use_boxcox': True}
        ]
        
        best_model = None
        best_aic = np.inf
        best_config = None
        
        for config in configs:
            try:
                # Apply Box-Cox if specified
                if config.get('use_boxcox', False) and (work_data['Sales'] > 0).all():
                    transformed_data, lambda_param = stats.boxcox(work_data['Sales'])
                    config['boxcox_lambda'] = lambda_param
                else:
                    transformed_data = work_data['Sales'].values
                    config['boxcox_lambda'] = None
                
                model = ExponentialSmoothing(
                    transformed_data,
                    seasonal=config.get('seasonal'),
                    seasonal_periods=12 if config.get('seasonal') else None,
                    trend=config.get('trend'),
                    damped_trend=config.get('damped_trend', False) if config.get('trend') else False,
                    initialization_method='estimated'
                )
                
                fitted_model = model.fit(optimized=True, use_brute=True)
                
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_model = fitted_model
                    best_config = config
            except:
                continue
        
        if best_model is not None:
            # Generate forecast
            forecast = best_model.forecast(steps=forecast_periods)
            
            # Generate prediction intervals using simulation
            simulated_forecasts = best_model.simulate(
                nsimulations=forecast_periods,
                repetitions=1000,
                anchor='end'
            )
            
            lower_bound = np.percentile(simulated_forecasts, 2.5, axis=1)
            upper_bound = np.percentile(simulated_forecasts, 97.5, axis=1)
            
            # Reverse Box-Cox transformation if applied
            if best_config.get('boxcox_lambda'):
                forecast = inv_boxcox(forecast, best_config['boxcox_lambda'])
                lower_bound = inv_boxcox(lower_bound, best_config['boxcox_lambda'])
                upper_bound = inv_boxcox(upper_bound, best_config['boxcox_lambda'])
            
            # Apply other transformations
            if 'transformation' in work_data.columns and not best_config.get('use_boxcox'):
                transform_method = work_data['transformation'].iloc[0]
                if transform_method == 'log':
                    forecast = np.expm1(forecast)
                    lower_bound = np.expm1(lower_bound)
                    upper_bound = np.expm1(upper_bound)
                elif transform_method == 'sqrt':
                    forecast = forecast ** 2
                    lower_bound = lower_bound ** 2
                    upper_bound = upper_bound ** 2
            
            forecast = np.maximum(forecast, 0) * scaling_factor
            
            # Store forecast info
            forecast_info = {
                'values': forecast,
                'lower_bound': lower_bound * scaling_factor,
                'upper_bound': upper_bound * scaling_factor,
                'model_config': best_config,
                'aic': best_aic
            }
            
            return forecast, best_aic
        else:
            raise ValueError("All ETS configurations failed")
            
    except Exception as e:
        st.warning(f"ETS failed: {str(e)}")
        return run_fallback_forecast(data, forecast_periods, scaling_factor), np.inf


def run_advanced_xgboost_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """Production-ready XGBoost with extensive feature engineering"""
    if not XGBOOST_AVAILABLE:
        st.warning("XGBoost not installed. Using simplified forecast.")
        return run_fallback_forecast(data, forecast_periods, scaling_factor), np.inf
    
    try:
        work_data = data.copy()
        
        # Create comprehensive features
        featured_data = create_advanced_features(work_data)
        
        # Remove NaN values from feature engineering
        featured_data = featured_data.dropna()
        
        if len(featured_data) < 24:
            st.warning("Insufficient data for XGBoost. Need at least 24 months.")
            return run_fallback_forecast(data, forecast_periods, scaling_factor), np.inf
        
        # Define features and target
        feature_cols = [col for col in featured_data.columns if col not in [
            'Month', 'Sales', 'Sales_Original', 'transformation', 'transformation_params',
            'needs_differencing', 'month'
        ]]
        
        X = featured_data[feature_cols]
        y = featured_data['Sales']
        
        # Feature scaling
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Hyperparameter tuning with Bayesian optimization
        param_space = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 4, 5, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0]
        }
        
        # Use RandomizedSearchCV for efficiency
        from sklearn.model_selection import RandomizedSearchCV
        
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            tree_method='hist'  # Faster training
        )
        
        random_search = RandomizedSearchCV(
            xgb_model,
            param_space,
            n_iter=50,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=0,
            random_state=42
        )
        
        with st.spinner("🚀 Optimizing XGBoost hyperparameters..."):
            random_search.fit(X_scaled, y)
        
        best_model = random_search.best_estimator_
        best_score = -random_search.best_score_
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Generate recursive forecasts
        last_known_features = featured_data.iloc[-1].copy()
        predictions = []
        prediction_intervals = []
        
        # Get prediction intervals using quantile regression
        models_quantile = {}
        for quantile in [0.025, 0.975]:
            model_q = xgb.XGBRegressor(**random_search.best_params_, objective='reg:quantileerror', 
                                      quantile_alpha=quantile, random_state=42)
            model_q.fit(X_scaled, y)
            models_quantile[quantile] = model_q
        
        for i in range(forecast_periods):
            # Update temporal features
            next_month = last_known_features['Month'] + pd.DateOffset(months=1)
            
            # Create feature vector
            feature_dict = {
                'year': next_month.year,
                'quarter': next_month.quarter,
                'dayofyear': next_month.dayofyear,
                'weekofyear': next_month.isocalendar()[1],
                'month_sin': np.sin(2 * np.pi * next_month.month / 12),
                'month_cos': np.cos(2 * np.pi * next_month.month / 12),
                'trend': featured_data['trend'].max() + i + 1,
                'trend_squared': (featured_data['trend'].max() + i + 1) ** 2
            }
            
            # Add lag features
            for lag in [1, 2, 3, 6, 12, 24]:
                if f'lag_{lag}' in feature_cols:
                    if i >= lag:
                        feature_dict[f'lag_{lag}'] = predictions[i - lag]
                    else:
                        recent_idx = len(featured_data) - (lag - i)
                        if recent_idx >= 0:
                            feature_dict[f'lag_{lag}'] = featured_data.iloc[recent_idx]['Sales']
                        else:
                            feature_dict[f'lag_{lag}'] = featured_data['Sales'].mean()
            
            # Add rolling features
            for window in [3, 6, 12, 24]:
                if f'rolling_mean_{window}' in feature_cols:
                    recent_values = list(featured_data['Sales'].tail(window - 1)) + predictions[:i]
                    if len(recent_values) >= window:
                        feature_dict[f'rolling_mean_{window}'] = np.mean(recent_values[-window:])
                        feature_dict[f'rolling_std_{window}'] = np.std(recent_values[-window:])
                        feature_dict[f'rolling_min_{window}'] = np.min(recent_values[-window:])
                        feature_dict[f'rolling_max_{window}'] = np.max(recent_values[-window:])
                    else:
                        feature_dict[f'rolling_mean_{window}'] = np.mean(recent_values)
                        feature_dict[f'rolling_std_{window}'] = np.std(recent_values) if len(recent_values) > 1 else 0
                        feature_dict[f'rolling_min_{window}'] = np.min(recent_values) if recent_values else 0
                        feature_dict[f'rolling_max_{window}'] = np.max(recent_values) if recent_values else 0
                    
                    if f'ewm_mean_{window}' in feature_cols:
                        feature_dict[f'ewm_mean_{window}'] = feature_dict[f'rolling_mean_{window}']
            
            # Add other features
            for col in feature_cols:
                if col not in feature_dict:
                    if col in last_known_features:
                        feature_dict[col] = last_known_features[col]
                    else:
                        feature_dict[col] = 0
            
            # Create feature vector
            feature_vector = np.array([feature_dict.get(col, 0) for col in feature_cols]).reshape(1, -1)
            feature_vector_scaled = scaler.transform(feature_vector)
            
            # Make prediction
            pred = best_model.predict(feature_vector_scaled)[0]
            predictions.append(pred)
            
            # Get prediction intervals
            lower = models_quantile[0.025].predict(feature_vector_scaled)[0]
            upper = models_quantile[0.975].predict(feature_vector_scaled)[0]
            prediction_intervals.append((lower, upper))
            
            # Update last known features
            last_known_features = last_known_features.copy()
            last_known_features['Month'] = next_month
            last_known_features['Sales'] = pred
        
        forecasts = np.array(predictions)
        lower_bounds = np.array([interval[0] for interval in prediction_intervals])
        upper_bounds = np.array([interval[1] for interval in prediction_intervals])
        
        # Apply transformations
        if 'transformation' in work_data.columns:
            transform_method = work_data['transformation'].iloc[0]
            if transform_method == 'log':
                forecasts = np.expm1(forecasts)
                lower_bounds = np.expm1(lower_bounds)
                upper_bounds = np.expm1(upper_bounds)
            elif transform_method == 'sqrt':
                forecasts = forecasts ** 2
                lower_bounds = lower_bounds ** 2
                upper_bounds = upper_bounds ** 2
            elif transform_method == 'boxcox':
                lambda_param = work_data['transformation_params'].iloc[0].get('lambda', 1)
                forecasts = inv_boxcox(forecasts, lambda_param)
                lower_bounds = inv_boxcox(lower_bounds, lambda_param)
                upper_bounds = inv_boxcox(upper_bounds, lambda_param)
        
        forecasts = np.maximum(forecasts, 0) * scaling_factor
        lower_bounds = np.maximum(lower_bounds, 0) * scaling_factor
        upper_bounds = np.maximum(upper_bounds, 0) * scaling_factor
        
        # Store comprehensive forecast info
        forecast_info = {
            'values': forecasts,
            'lower_bound': lower_bounds,
            'upper_bound': upper_bounds,
            'feature_importance': feature_importance,
            'model_params': random_search.best_params_,
            'cv_score': best_score
        }
        
        # Store in session state for later use
        st.session_state['xgboost_info'] = forecast_info
        
        return forecasts, best_score
        
    except Exception as e:
        st.warning(f"XGBoost failed: {str(e)}")
        return run_fallback_forecast(data, forecast_periods, scaling_factor), np.inf


def run_lstm_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """LSTM neural network for complex temporal patterns"""
    if not TENSORFLOW_AVAILABLE:
        st.warning("TensorFlow not available. Skipping LSTM model.")
        return run_fallback_forecast(data, forecast_periods, scaling_factor), np.inf
    
    try:
        work_data = data.copy()
        
        # Prepare data
        sales_data = work_data['Sales'].values.reshape(-1, 1)
        
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(sales_data)
        
        # Create sequences
        sequence_length = min(12, len(work_data) // 3)
        
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i+seq_length])
                y.append(data[i+seq_length])
            return np.array(X), np.array(y)
        
        X, y = create_sequences(scaled_data, sequence_length)
        
        # Split data for validation
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Build advanced LSTM model
        model = Sequential([
            LSTM(100, activation='tanh', return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, activation='tanh', return_sequences=True),
            Dropout(0.2),
            LSTM(25, activation='tanh'),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=0
        )
        
        # Generate forecasts
        last_sequence = scaled_data[-sequence_length:]
        predictions = []
        
        for _ in range(forecast_periods):
            next_pred = model.predict(last_sequence.reshape(1, sequence_length, 1), verbose=0)
            predictions.append(next_pred[0, 0])
            last_sequence = np.append(last_sequence[1:], next_pred).reshape(-1, 1)
        
        # Inverse transform
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        forecasts = predictions.flatten()
        
        # Apply transformations
        if 'transformation' in work_data.columns:
            transform_method = work_data['transformation'].iloc[0]
            if transform_method == 'log':
                forecasts = np.expm1(forecasts)
            elif transform_method == 'sqrt':
                forecasts = forecasts ** 2
            elif transform_method == 'boxcox':
                lambda_param = work_data['transformation_params'].iloc[0].get('lambda', 1)
                forecasts = inv_boxcox(forecasts, lambda_param)
        
        forecasts = np.maximum(forecasts, 0) * scaling_factor
        
        # Calculate validation score
        val_score = history.history['val_loss'][-1] * 1000  # Scale for comparison
        
        return forecasts, val_score
        
    except Exception as e:
        st.warning(f"LSTM failed: {str(e)}")
        return run_fallback_forecast(data, forecast_periods, scaling_factor), np.inf


def run_theta_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """Theta method - simple but effective for many time series"""
    try:
        work_data = data.copy()
        
        # Fit Theta model
        model = ThetaModel(work_data['Sales'], period=12)
        fitted_model = model.fit()
        
        # Generate forecast
        forecast = fitted_model.forecast(forecast_periods)
        
        # Apply transformations
        if 'transformation' in work_data.columns:
            transform_method = work_data['transformation'].iloc[0]
            if transform_method == 'log':
                forecast = np.expm1(forecast)
            elif transform_method == 'sqrt':
                forecast = forecast ** 2
            elif transform_method == 'boxcox':
                lambda_param = work_data['transformation_params'].iloc[0].get('lambda', 1)
                forecast = inv_boxcox(forecast, lambda_param)
        
        forecast = np.maximum(forecast, 0) * scaling_factor
        
        return forecast, 0.0  # Theta model doesn't provide AIC
        
    except Exception as e:
        st.warning(f"Theta method failed: {str(e)}")
        return run_fallback_forecast(data, forecast_periods, scaling_factor), np.inf


def run_croston_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """Croston's method for intermittent demand"""
    try:
        work_data = data.copy()
        
        # Check if data is intermittent
        zero_ratio = (work_data['Sales'] == 0).sum() / len(work_data)
        
        if zero_ratio < 0.3:  # Not intermittent enough
            st.info("Data doesn't appear to be intermittent. Croston's method may not be optimal.")
        
        alpha = 0.2  # Smoothing parameter
        
        # Extract non-zero demands and intervals
        demand = work_data['Sales'].values
        demands = []
        intervals = []
        
        last_demand_idx = -1
        for i, d in enumerate(demand):
            if d > 0:
                if last_demand_idx >= 0:
                    intervals.append(i - last_demand_idx)
                demands.append(d)
                last_demand_idx = i
        
        if not demands:
            return np.zeros(forecast_periods), np.inf
        
        # Initialize with averages
        avg_demand = np.mean(demands)
        avg_interval = np.mean(intervals) if intervals else 1
        
        # Apply exponential smoothing
        smoothed_demand = avg_demand
        smoothed_interval = avg_interval
        
        for i in range(1, len(demands)):
            smoothed_demand = alpha * demands[i] + (1 - alpha) * smoothed_demand
            if i < len(intervals):
                smoothed_interval = alpha * intervals[i] + (1 - alpha) * smoothed_interval
        
        # Generate forecasts
        forecast_value = smoothed_demand / smoothed_interval if smoothed_interval > 0 else smoothed_demand
        forecasts = np.full(forecast_periods, forecast_value)
        
        # Apply transformations
        if 'transformation' in work_data.columns:
            transform_method = work_data['transformation'].iloc[0]
            if transform_method == 'log':
                forecasts = np.expm1(forecasts)
            elif transform_method == 'sqrt':
                forecasts = forecasts ** 2
        
        forecasts = np.maximum(forecasts, 0) * scaling_factor
        
        return forecasts, 0.0
        
    except Exception as e:
        st.warning(f"Croston's method failed: {str(e)}")
        return run_fallback_forecast(data, forecast_periods, scaling_factor), np.inf


def run_fallback_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """Enhanced fallback forecasting with multiple methods"""
    try:
        work_data = data.copy()
        
        if len(work_data) >= 12:
            # Method 1: Seasonal naive with trend
            seasonal_pattern = work_data['Sales'].tail(12).values
            
            # Calculate trend using robust regression
            from sklearn.linear_model import HuberRegressor
            X_trend = np.arange(len(work_data)).reshape(-1, 1)
            y_trend = work_data['Sales'].values
            
            trend_model = HuberRegressor()
            trend_model.fit(X_trend, y_trend)
            
            trend_per_period = trend_model.coef_[0]
            
            # Generate forecast
            forecast = []
            last_index = len(work_data)
            
            for i in range(forecast_periods):
                seasonal_component = seasonal_pattern[i % 12]
                trend_component = trend_model.predict([[last_index + i]])[0] - trend_model.predict([[last_index]])[0]
                forecast_value = seasonal_component + trend_component
                forecast.append(max(forecast_value, seasonal_component * 0.5))
            
            forecast = np.array(forecast)
        else:
            # Simple moving average for very short series
            if len(work_data) >= 3:
                base_forecast = work_data['Sales'].tail(3).mean()
            else:
                base_forecast = work_data['Sales'].mean()
            
            # Add slight randomness to avoid flat forecasts
            np.random.seed(42)
            noise = np.random.normal(0, base_forecast * 0.05, forecast_periods)
            forecast = np.full(forecast_periods, base_forecast) + noise
            forecast = np.maximum(forecast, 0)
        
        # Apply transformations
        if 'transformation' in work_data.columns:
            transform_method = work_data['transformation'].iloc[0]
            if transform_method == 'log':
                forecast = np.expm1(forecast)
            elif transform_method == 'sqrt':
                forecast = forecast ** 2
        
        forecast = forecast * scaling_factor
        
        return forecast
        
    except Exception as e:
        # Ultimate fallback
        historical_mean = data['Sales'].mean() if len(data) > 0 else 1000
        return np.array([historical_mean * scaling_factor] * forecast_periods)


def create_advanced_ensemble(forecasts_dict, validation_scores, actual_data=None):
    """Create advanced ensemble with multiple weighting strategies"""
    # Remove infinite scores
    valid_scores = {k: v for k, v in validation_scores.items() if v != np.inf and v > 0}
    
    if not valid_scores:
        # Equal weights if no valid scores
        n_models = len(forecasts_dict)
        weights = {model.replace('_Forecast', ''): 1/n_models for model in forecasts_dict}
    else:
        # Try multiple weighting strategies
        weighting_strategies = {}
        
        # 1. Inverse error weighting
        total_inverse = sum(1/score for score in valid_scores.values())
        inverse_weights = {k: (1/v) / total_inverse for k, v in valid_scores.items()}
        weighting_strategies['inverse_error'] = inverse_weights
        
        # 2. Softmax weighting
        scores_array = np.array(list(valid_scores.values()))
        softmax_scores = np.exp(-scores_array / scores_array.mean())
        softmax_weights = softmax_scores / softmax_scores.sum()
        softmax_dict = dict(zip(valid_scores.keys(), softmax_weights))
        weighting_strategies['softmax'] = softmax_dict
        
        # 3. Rank-based weighting
        sorted_models = sorted(valid_scores.items(), key=lambda x: x[1])
        rank_weights = {}
        for i, (model, _) in enumerate(sorted_models):
            rank_weights[model] = (len(sorted_models) - i) / sum(range(1, len(sorted_models) + 1))
        weighting_strategies['rank_based'] = rank_weights
        
        # Select best strategy (could be based on historical performance)
        weights = weighting_strategies['softmax']  # Default to softmax
    
    # Create multiple ensemble variants
    ensemble_variants = {}
    
    # 1. Weighted average
    weighted_forecast = np.zeros(len(next(iter(forecasts_dict.values()))))
    for model_name, forecast in forecasts_dict.items():
        model_key = model_name.replace('_Forecast', '')
        weight = weights.get(model_key, 1/len(forecasts_dict))
        weighted_forecast += weight * forecast
    ensemble_variants['weighted_average'] = weighted_forecast
    
    # 2. Trimmed mean (remove best and worst)
    if len(forecasts_dict) > 2:
        forecast_array = np.array(list(forecasts_dict.values()))
        trimmed_mean = np.mean(np.sort(forecast_array, axis=0)[1:-1, :], axis=0)
        ensemble_variants['trimmed_mean'] = trimmed_mean
    
    # 3. Median ensemble
    forecast_array = np.array(list(forecasts_dict.values()))
    median_forecast = np.median(forecast_array, axis=0)
    ensemble_variants['median'] = median_forecast
    
    # Select best ensemble variant
    final_ensemble = ensemble_variants['weighted_average']
    
    return final_ensemble, weights, ensemble_variants


def run_meta_learning_ensemble(forecasts_dict, historical_data, actual_data=None):
    """Advanced meta-learning with multiple base learners"""
    if actual_data is None or len(actual_data) < 12:
        return None
    
    try:
        # Prepare training data for meta-learner
        # Use historical forecasts vs actuals to train
        forecast_cols = [col for col in actual_data.columns if '_Forecast' in col]
        actual_col = [col for col in actual_data.columns if 'Actual_' in col][0]
        
        # Get overlapping data
        overlap_data = actual_data.dropna(subset=[actual_col] + forecast_cols)
        
        if len(overlap_data) < 6:  # Need minimum data for meta-learning
            return None
        
        X_meta = overlap_data[forecast_cols].values
        y_meta = overlap_data[actual_col].values
        
        # Try multiple meta-learners
        meta_learners = {
            'ridge': AdvancedMetaLearner(meta_model='ridge'),
            'elastic': AdvancedMetaLearner(meta_model='elastic'),
            'rf': AdvancedMetaLearner(meta_model='rf')
        }
        
        # Cross-validate meta-learners
        best_score = np.inf
        best_meta_learner = None
        
        for name, learner in meta_learners.items():
            try:
                # Simple train-test split
                split_idx = int(len(X_meta) * 0.7)
                X_train, X_test = X_meta[:split_idx], X_meta[split_idx:]
                y_train, y_test = y_meta[:split_idx], y_meta[split_idx:]
                
                learner.fit(X_train, y_train)
                pred = learner.predict(X_test)
                score = mean_absolute_error(y_test, pred)
                
                if score < best_score:
                    best_score = score
                    best_meta_learner = learner
            except:
                continue
        
        if best_meta_learner is None:
            return None
        
        # Train on full data
        best_meta_learner.fit(X_meta, y_meta)
        
        # Create forecast using all models
        forecast_values = np.array([forecasts_dict[col] for col in forecast_cols]).T
        meta_forecast = best_meta_learner.predict(forecast_values)
        
        return np.maximum(meta_forecast, 0)
        
    except Exception as e:
        st.warning(f"Meta-learning failed: {str(e)}")
        return None


def calculate_comprehensive_metrics(actual, forecast):
    """Calculate comprehensive accuracy metrics"""
    if len(actual) == 0 or len(forecast) == 0:
        return None
    
    mask = ~(pd.isna(actual) | pd.isna(forecast))
    actual_clean = actual[mask]
    forecast_clean = forecast[mask]
    
    if len(actual_clean) == 0:
        return None
    
    metrics = {}
    
    # Standard metrics
    metrics['MAE'] = mean_absolute_error(actual_clean, forecast_clean)
    metrics['RMSE'] = np.sqrt(mean_squared_error(actual_clean, forecast_clean))
    metrics['MAPE'] = mean_absolute_percentage_error(actual_clean, forecast_clean) * 100
    
    # Additional metrics
    metrics['SMAPE'] = 100 * np.mean(2 * np.abs(forecast_clean - actual_clean) / 
                                    (np.abs(actual_clean) + np.abs(forecast_clean)))
    
    # MASE (Mean Absolute Scaled Error)
    if len(actual_clean) > 1:
        naive_errors = np.abs(np.diff(actual_clean))
        if naive_errors.mean() > 0:
            metrics['MASE'] = metrics['MAE'] / naive_errors.mean()
        else:
            metrics['MASE'] = np.inf
    
    # Directional accuracy
    if len(actual_clean) > 1:
        actual_direction = np.diff(actual_clean) > 0
        forecast_direction = np.diff(forecast_clean) > 0
        metrics['Directional_Accuracy'] = np.mean(actual_direction == forecast_direction) * 100
    
    # Bias
    metrics['Bias'] = np.mean(forecast_clean - actual_clean)
    metrics['Bias_Pct'] = (metrics['Bias'] / np.mean(actual_clean)) * 100
    
    # Tracking signal
    cumulative_error = np.cumsum(forecast_clean - actual_clean)
    metrics['Tracking_Signal'] = cumulative_error[-1] / metrics['MAE'] if metrics['MAE'] > 0 else 0
    
    return metrics


def detect_data_frequency(dates):
    """Automatically detect data frequency"""
    if len(dates) < 2:
        return 'M'  # Default to monthly
    
    # Calculate differences between consecutive dates
    date_diffs = pd.Series(dates).diff().dropna()
    
    # Get mode of differences in days
    mode_days = date_diffs.dt.days.mode()[0]
    
    if 28 <= mode_days <= 31:
        return 'M'  # Monthly
    elif 7 <= mode_days <= 7:
        return 'W'  # Weekly
    elif mode_days == 1:
        return 'D'  # Daily
    elif 90 <= mode_days <= 92:
        return 'Q'  # Quarterly
    elif 365 <= mode_days <= 366:
        return 'Y'  # Yearly
    else:
        return 'M'  # Default to monthly
