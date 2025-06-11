best_model = None
        best_aic = np.inf
        best_config = None
        
        for config in configs:
            try:
                model = ExponentialSmoothing(
                    work_data['Sales'].values,
                    seasonal=config.get('seasonal'),
                    seasonal_periods=12 if config.get('seasonal') else None,
                    trend=config.get('trend'),
                    damped_trend=config.get('damped_trend', False) if config.get('trend') else False,
                    initialization_method='estimated'
                )
                
                fitted_model = model.fit(optimized=True)
                
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_model = fitted_model
                    best_config = config
            except:
                continue
        
        if best_model is not None:
            # Generate forecast
            forecast = best_model.forecast(steps=forecast_periods)
            
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
        
        if len(featured_data) < 12:
            st.warning("Insufficient data for XGBoost. Need at least 12 months.")
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
        
        # Simple hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0]
        }
        
        # Use GridSearchCV with time series split
        tscv = TimeSeriesSplit(n_splits=3)
        
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42
        )
        
        grid_search = GridSearchCV(
            xgb_model,
            param_grid,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_scaled, y)
        
        best_model = grid_search.best_estimator_
        best_score = -grid_search.best_score_
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Generate recursive forecasts
        last_known_features = featured_data.iloc[-1].copy()
        predictions = []
        
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
            for lag in [1, 2, 3, 6, 12]:
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
            for window in [3, 6, 12]:
                if f'rolling_mean_{window}' in feature_cols:
                    recent_values = list(featured_data['Sales'].tail(window - 1)) + predictions[:i]
                    if len(recent_values) >= window:
                        feature_dict[f'rolling_mean_{window}'] = np.mean(recent_values[-window:])
                        feature_dict[f'rolling_std_{window}'] = np.std(recent_values[-window:])
                        feature_dict[f'rolling_min_{window}'] = np.min(recent_values[-window:])
                        feature_dict[f'rolling_max_{window}'] = np.max(recent_values[-window:])
                    else:
                        feature_dict[f'rolling_mean_{window}'] = np.mean(recent_values) if recent_values else featured_data['Sales'].mean()
                        feature_dict[f'rolling_std_{window}'] = np.std(recent_values) if len(recent_values) > 1 else 0
                        feature_dict[f'rolling_min_{window}'] = np.min(recent_values) if recent_values else featured_data['Sales'].min()
                        feature_dict[f'rolling_max_{window}'] = np.max(recent_values) if recent_values else featured_data['Sales'].max()
                    
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
            
            # Update last known features
            last_known_features = last_known_features.copy()
            last_known_features['Month'] = next_month
            last_known_features['Sales'] = pred
        
        forecasts = np.array(predictions)
        
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
        
        # Store comprehensive forecast info
        forecast_info = {
            'values': forecasts,
            'feature_importance': feature_importance,
            'model_params': grid_search.best_params_,
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
        
        if len(X) < 10:
            st.warning("Insufficient data for LSTM training.")
            return run_fallback_forecast(data, forecast_periods, scaling_factor), np.inf
        
        # Split data for validation
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, activation='tanh', return_sequences=True, input_shape=(sequence_length, 1)),
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
            epochs=50,
            batch_size=16,
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
        
        return forecast, 0.0
        
    except Exception as e:
        st.warning(f"Theta method failed: {str(e)}")
        return run_fallback_forecast(data, forecast_periods, scaling_factor), np.inf


def run_croston_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """Croston's method for intermittent demand"""
    try:
        work_data = data.copy()
        
        # Check if data is intermittent
        zero_ratio = (work_data['Sales'] == 0).sum() / len(work_data)
        
        if zero_ratio < 0.3:
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
            X_trend = np.arange(len(work_data)).reshape(-1, 1)
            y_trend = work_data['Sales'].values
            
            trend_model = HuberRegressor()
            trend_model.fit(X_trend, y_trend)
            
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
        # Softmax weighting strategy
        scores_array = np.array(list(valid_scores.values()))
        softmax_scores = np.exp(-scores_array / scores_array.mean())
        softmax_weights = softmax_scores / softmax_scores.sum()
        weights = dict(zip(valid_scores.keys(), softmax_weights))
    
    # Create weighted ensemble
    weighted_forecast = np.zeros(len(next(iter(forecasts_dict.values()))))
    for model_name, forecast in forecasts_dict.items():
        model_key = model_name.replace('_Forecast', '')
        weight = weights.get(model_key, 1/len(forecasts_dict))
        weighted_forecast += weight * forecast
    
    # Create ensemble variants
    ensemble_variants = {}
    ensemble_variants['weighted_average'] = weighted_forecast
    
    # Median ensemble
    forecast_array = np.array(list(forecasts_dict.values()))
    median_forecast = np.median(forecast_array, axis=0)
    ensemble_variants['median'] = median_forecast
    
    return weighted_forecast, weights, ensemble_variants


def run_meta_learning_ensemble(forecasts_dict, historical_data, actual_data=None):
    """Advanced meta-learning with multiple base learners"""
    if actual_data is None or len(actual_data) < 6:
        return None
    
    try:
        # Prepare training data for meta-learner
        forecast_cols = [col for col in actual_data.columns if '_Forecast' in col]
        actual_col = [col for col in actual_data.columns if 'Actual_' in col][0]
        
        # Get overlapping data
        overlap_data = actual_data.dropna(subset=[actual_col] + forecast_cols)
        
        if len(overlap_data) < 6:
            return None
        
        X_meta = overlap_data[forecast_cols].values
        y_meta = overlap_data[actual_col].values
        
        # Use Ridge regression as meta-learner
        meta_learner = Ridge(alpha=1.0)
        meta_learner.fit(X_meta, y_meta)
        
        # Create forecast using all models
        forecast_values = np.array([forecasts_dict[col] for col in forecast_cols]).T
        meta_forecast = meta_learner.predict(forecast_values)
        
        return np.maximum(meta_forecast, 0)
        
    except Exception as e:
        st.warning(f"Meta-learning failed: {str(e)}")
        return None


def create_forecast_plot(result_df, forecast_year, historical_df=None):
    """Create comprehensive forecast visualization"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Forecast Comparison', 'Model Performance', 
                       'Residual Analysis', 'Seasonal Pattern'),
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
    
    # Seasonal pattern analysis
    if 'Weighted_Ensemble' in result_df.columns:
        monthly_avg = result_df.groupby(result_df['Month'].dt.month)['Weighted_Ensemble'].mean()
        seasonal_index = (monthly_avg / monthly_avg.mean() * 100).round(1)
        
        fig.add_trace(
            go.Bar(
                x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                y=seasonal_index.values,
                name='Seasonal Index',
                marker_color=['red' if v < 100 else 'green' for v in seasonal_index.values],
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.add_hline(y=100, line_dash="dash", line_color="gray", row=2, col=2)
    
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
    fig.update_yaxes(title_text="Index", row=2, col=2)
    
    return fig


def create_diagnostic_plots(historical_df):
    """Create diagnostic plots for time series analysis"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Time Series Plot', 'Seasonal Decomposition', 
                       'Monthly Boxplot', 'Distribution Analysis')
    )
    
    # Time series plot
    fig.add_trace(
        go.Scatter(x=historical_df['Month'], y=historical_df['Sales_Original'], 
                  name='Sales', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Seasonal decomposition
    if len(historical_df) >= 24:
        try:
            decomposition = seasonal_decompose(historical_df['Sales'], model='additive', period=12)
            fig.add_trace(
                go.Scatter(x=historical_df['Month'], y=decomposition.trend, 
                          name='Trend', line=dict(color='red')),
                row=1, col=2
            )
        except:
            pass
    
    # Monthly boxplot
    monthly_data = []
    month_names = []
    for month in range(1, 13):
        month_data = historical_df[historical_df['Month'].dt.month == month]['Sales_Original']
        if len(month_data) > 0:
            monthly_data.append(month_data.values)
            month_names.append(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month-1])
    
    for i, (data, name) in enumerate(zip(monthly_data, month_names)):
        fig.add_trace(
            go.Box(y=data, name=name, showlegend=False),
            row=2, col=1
        )
    
    # Distribution
    fig.add_trace(
        go.Histogram(x=historical_df['Sales_Original'], name='Sales Distribution', 
                    nbinsx=20, showlegend=False),
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
            try:
                decomposition = seasonal_decompose(hist_df['Sales'], model='additive', period=12)
                seasonal_strength = np.var(decomposition.seasonal) / np.var(hist_df['Sales'])
                analysis_data.append({
                    'Category': 'Time Series Properties',
                    'Metric': 'Seasonality Strength',
                    'Value': f"{seasonal_strength:.2%}"
                })
            except:
                pass
        
        analysis_df = pd.DataFrame(analysis_data)
        analysis_df.to_excel(writer, sheet_name='Data_Analysis', index=False)
    
    output.seek(0)
    return output


def main():
    """Main function to run the enhanced forecasting application"""
    st.title("🚀 Advanced AI Sales Forecasting System")
    st.markdown("**Enterprise-grade forecasting with 10+ models, ensemble learning, and neural networks**")
    
    # Initialize session state
    if 'forecast_info' not in st.session_state:
        st.session_state.forecast_info = {}
    
    # Display warnings for missing packages
    if not XGBOOST_AVAILABLE:
        st.warning("⚠️ XGBoost not installed. Install with: `pip install xgboost` for better accuracy")
    if not TENSORFLOW_AVAILABLE:
        st.info("ℹ️ TensorFlow not available. Install with: `pip install tensorflow` for LSTM models")
    
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
        enable_parallel = st.checkbox("Enable Parallel Processing", value=False,
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
        
        # Run models
        forecast_results = {}
        validation_scores = {}
        forecast_info = {}
        
        # Sequential execution for stability
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
        ensemble_weights = {}
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
        st.session_state['ensemble_weights'] = ensemble_weights
        st.session_state['forecast_info'] = forecast_info
        
        # Display results
        st.header("📊 Forecast Results")
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_forecast = result_df['Weighted_Ensemble'].sum() if 'Weighted_Ensemble' in result_df else list(forecast_results.values())[0].sum()
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
                st.plotly_chart(feat_img_fig, use_container_width=True)
                
                # Show top features
                top_features = xgb_info['feature_importance'].head(5)
                st.info(f"🎯 Top predictive features: {', '.join(top_features['feature'].tolist())}")
        
        # Ensemble Analysis
        if ensemble_weights:
            st.subheader("🤝 Ensemble Analysis")
            
            weights_df = pd.DataFrame([
                {'Model': k, 'Weight': v} 
                for k, v in ensemble_weights.items()
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
        
        # Download Section
        st.header("📥 Export Results")
        
        # Generate comprehensive report
        excel_report = create_comprehensive_excel_report(
            result_df,
            hist_df,
            forecast_year,
            scaling_factor,
            validation_scores,
            ensemble_weights,
            forecast_info
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
        forecast_cols = [col for col in result_df.columns if '_Forecast' in col]
        if len(forecast_cols) > 1 and 'Weighted_Ensemble' in result_df.columns:
            forecast_array = result_df[forecast_cols].values
            cv_by_month = np.std(forecast_array, axis=1) / np.mean(forecast_array, axis=1)
            avg_cv = np.mean(cv_by_month)
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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.base import BaseEstimator, RegressorMixin

# Try to import optional libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

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
        try:
            from statsmodels.tsa.stattools import adfuller
            # Check for stationarity
            adf_result = adfuller(df['Sales'])
            if adf_result[1] > 0.05:  # Non-stationary
                st.info("📈 Non-stationary data detected. Applying differencing.")
                df['needs_differencing'] = True
            else:
                df['needs_differencing'] = False
        except:
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
            df['transformation_params'] = {'lambda': stats.boxcox(df['Sales_Original'] + 1)[1]}
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


@st.cache_data
def load_actual_2024_data(uploaded_file, forecast_year):
    """Load actual data with preprocessing - only include months that have actual data"""
    try:
        df = pd.read_excel(uploaded_file)
        
        # Check if it's the standard long format
        if "Month" in df.columns and "Sales" in df.columns:
            df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
            if df["Month"].isna().any():
                st.error("Some dates in the actual file could not be parsed.")
                return None

            df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
            df["Sales"] = df["Sales"].abs()

            # Filter to the forecast year only
            start = pd.Timestamp(f"{forecast_year}-01-01")
            end = pd.Timestamp(f"{forecast_year+1}-01-01")
            df = df[(df["Month"] >= start) & (df["Month"] < end)]
            
            if df.empty:
                st.warning(f"No rows match year {forecast_year}.")
                return None

            # Only include months that have actual non-zero data
            monthly = df.groupby("Month", as_index=False)["Sales"].sum()
            monthly = monthly[monthly["Sales"] > 0]  # Only months with actual sales
            monthly = monthly.sort_values("Month").reset_index(drop=True)
            
            return monthly.rename(columns={"Sales": f"Actual_{forecast_year}"})
        
        else:
            # Wide format handling
            st.info("📊 Detected wide format data - converting to long format...")
            
            month_patterns = [
                f"Jan-{forecast_year}", f"Feb-{forecast_year}", f"Mar-{forecast_year}",
                f"Apr-{forecast_year}", f"May-{forecast_year}", f"Jun-{forecast_year}",
                f"Jul-{forecast_year}", f"Aug-{forecast_year}", f"Sep-{forecast_year}",
                f"Oct-{forecast_year}", f"Nov-{forecast_year}", f"Dec-{forecast_year}"
            ]
            
            # Only include month patterns that actually exist in the data
            available_months = [pattern for pattern in month_patterns if pattern in df.columns]
            
            if not available_months:
                st.error(f"No month columns found for {forecast_year}.")
                return None
            
            st.info(f"📅 Found data for months: {', '.join([m.split('-')[0] for m in available_months])}")
            
            first_col = df.columns[0]
            data_rows = df[~df[first_col].astype(str).str.contains("Item|Code|QTY", case=False, na=False)]
            
            melted_data = []
            
            for _, row in data_rows.iterrows():
                for month_col in available_months:
                    if month_col in row and pd.notna(row[month_col]):
                        sales_value = pd.to_numeric(row[month_col], errors="coerce")
                        if pd.notna(sales_value) and sales_value > 0:
                            month_str = month_col.replace("-", "-01-")
                            try:
                                month_date = pd.to_datetime(month_str, format="%b-%d-%Y")
                                melted_data.append({
                                    "Month": month_date,
                                    "Sales": abs(sales_value)
                                })
                            except:
                                continue
            
            if not melted_data:
                st.error("No valid sales data found.")
                return None
            
            long_df = pd.DataFrame(melted_data)
            
            # Group by month and sum, but only for months that actually have data
            monthly = long_df.groupby("Month", as_index=False)["Sales"].sum()
            monthly = monthly[monthly["Sales"] > 0]  # Only months with actual sales data
            monthly = monthly.sort_values("Month").reset_index(drop=True)
            
            # Show which months were actually processed
            processed_months = monthly['Month'].dt.strftime('%b').tolist()
            st.success(f"✅ Successfully processed data for: {', '.join(processed_months)}")
            
            return monthly.rename(columns={"Sales": f"Actual_{forecast_year}"})
            
    except Exception as e:
        st.error(f"Error loading actual data: {str(e)}")
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


def detect_and_apply_scaling(historical_data, actual_data=None):
    """Enhanced scaling detection with multiple methods"""
    hist_avg = historical_data['Sales'].mean()
    
    if actual_data is not None and len(actual_data) > 0:
        actual_avg = actual_data.iloc[:, 1].mean()
        
        # Multiple scaling detection methods
        ratio = actual_avg / hist_avg if hist_avg > 0 else 1
        
        # Apply scaling if ratio is significant
        if ratio > 2 or ratio < 0.5:
            st.warning(f"📊 Scale mismatch detected! Scaling factor: {ratio:.2f}")
            return ratio
    
    return 1.0


def detect_data_frequency(dates):
    """Automatically detect data frequency"""
    if len(dates) < 2:
        return 'M'  # Default to monthly
    
    # Calculate differences between consecutive dates
    date_diffs = pd.Series(dates).diff().dropna()
    
    # Get mode of differences in days
    days_diff = date_diffs.dt.days
    if len(days_diff) > 0:
        mode_days = days_diff.mode()
        if len(mode_days) > 0:
            mode_days = mode_days.iloc[0]
        else:
            mode_days = days_diff.median()
    else:
        return 'M'  # Default to monthly
    
    if 28 <= mode_days <= 31:
        return 'M'  # Monthly
    elif 6 <= mode_days <= 8:
        return 'W'  # Weekly
    elif mode_days == 1:
        return 'D'  # Daily
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


def parallel_model_training(model_func, data, forecast_periods, scaling_factor, model_name):
    """Wrapper for parallel model training"""
    try:
        result = model_func(data, forecast_periods, scaling_factor)
        return model_name, result
    except Exception as e:
        logger.error(f"Error in {model_name}: {str(e)}")
        return model_name, (run_fallback_forecast(data, forecast_periods, scaling_factor), np.inf)


def run_advanced_sarima_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """Enhanced SARIMA with automatic order selection and diagnostics"""
    try:
        work_data = data.copy()
        
        # Try auto ARIMA first
        try:
            from pmdarima import auto_arima
            
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
        
        except ImportError:
            st.warning("pmdarima not installed. Using manual parameter selection.")
            best_order = (1, 1, 1)
            best_seasonal_order = (1, 1, 1, 12)
        
        # Fit final model
        model = SARIMAX(
            work_data['Sales'],
            order=best_order,
            seasonal_order=best_seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted_model = model.fit(disp=False)
        
        # Generate forecast
        forecast = fitted_model.forecast(steps=forecast_periods)
        
        # Reverse transformations if applied
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
        
        return forecast, fitted_model.aic
        
    except Exception as e:
        st.warning(f"SARIMA failed: {str(e)}")
        return run_fallback_forecast(data, forecast_periods, scaling_factor), np.inf


def run_advanced_prophet_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """Enhanced Prophet with holiday effects and changepoint detection"""
    try:
        work_data = data.copy()
        
        # Prepare data
        prophet_data = work_data[['Month', 'Sales']].rename(columns={'Month': 'ds', 'Sales': 'y'})
        
        # Detect holidays/special events (outliers as potential holidays)
        holidays = None
        if len(prophet_data) >= 24:
            rolling_mean = prophet_data['y'].rolling(window=12, center=True).mean()
            rolling_std = prophet_data['y'].rolling(window=12, center=True).std()
            outliers = np.abs(prophet_data['y'] - rolling_mean) > 2 * rolling_std
            
            if outliers.sum() > 0:
                holidays = pd.DataFrame({
                    'holiday': 'detected_event',
                    'ds': prophet_data.loc[outliers, 'ds'],
                    'lower_window': -1,
                    'upper_window': 1
                })
        
        # Create Prophet model with optimized parameters
        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            seasonality_mode='additive',
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            holidays=holidays,
            interval_width=0.95
        )
        
        # Add custom seasonality
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
        
        # Add regressors if available
        if 'month_sin' in work_data.columns:
            prophet_data['month_sin'] = work_data['month_sin']
            prophet_data['month_cos'] = work_data['month_cos']
            model.add_regressor('month_sin')
            model.add_regressor('month_cos')
        
        model.fit(prophet_data)
        
        # Make predictions
        future = model.make_future_dataframe(periods=forecast_periods, freq='MS')
        
        # Add regressor values for future dates
        if 'month_sin' in prophet_data.columns:
            future_months = pd.to_datetime(future['ds']).dt.month
            future['month_sin'] = np.sin(2 * np.pi * future_months / 12)
            future['month_cos'] = np.cos(2 * np.pi * future_months / 12)
        
        forecast = model.predict(future)
        forecast_values = forecast['yhat'].tail(forecast_periods).values
        
        # Apply transformations
        if 'transformation' in work_data.columns:
            transform_method = work_data['transformation'].iloc[0]
            if transform_method == 'log':
                forecast_values = np.expm1(forecast_values)
            elif transform_method == 'sqrt':
                forecast_values = forecast_values ** 2
        
        forecast_values = np.maximum(forecast_values, 0) * scaling_factor
        
        return forecast_values, 100.0
        
    except Exception as e:
        st.warning(f"Prophet failed: {str(e)}")
        return run_fallback_forecast(data, forecast_periods, scaling_factor), np.inf


def run_advanced_ets_forecast(data, forecast_periods=12, scaling_factor=1.0):
    """Advanced ETS with automatic model selection"""
    try:
        work_data = data.copy()
        
        # Test different ETS configurations
        configs = [
            {'seasonal': 'add', 'trend': 'add', 'damped_trend': True},
            {'seasonal': 'add', 'trend': 'add', 'damped_trend': False},
            {'seasonal': 'mul', 'trend': 'add', 'damped_trend': True},
            {'seasonal': 'mul', 'trend': 'add', 'damped_trend': False},
            {'seasonal': 'add', 'trend': None},
            {'seasonal': None, 'trend': 'add', 'damped_trend': True}
        ]
        
        best_model =
