import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class SparePartsPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.predictions = {}
        self.actual_data = {}
        self.training_data = {}
        
    def load_data(self, file_path, year):
        """
        Load spare parts data from Excel/CSV file
        Expected format: Item Code in column A, monthly data in subsequent columns
        """
        try:
            # Try reading as Excel first, then CSV
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
            
            print(f"Loaded data for {year}: {df.shape[0]} parts, {df.shape[1]} columns")
            
            # Clean column names - remove extra spaces and standardize
            df.columns = df.columns.str.strip()
            
            # Store the data
            self.training_data[year] = df
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def prepare_features(self, df):
        """
        Prepare features for prediction from monthly sales data
        """
        # Identify month columns (assuming they contain 'QTY' or are numeric columns after item code)
        item_col = df.columns[0]  # First column is item code
        month_cols = [col for col in df.columns[1:] if 'QTY' in str(col) or df[col].dtype in ['int64', 'float64']]
        
        features_list = []
        
        for idx, row in df.iterrows():
            item_code = row[item_col]
            monthly_sales = [row[col] for col in month_cols if pd.notna(row[col])]
            
            if len(monthly_sales) >= 3:  # Need at least 3 months of data
                # Create features
                features = {
                    'item_code': item_code,
                    'total_sales': sum(monthly_sales),
                    'avg_monthly_sales': np.mean(monthly_sales),
                    'max_monthly_sales': max(monthly_sales),
                    'min_monthly_sales': min(monthly_sales),
                    'sales_std': np.std(monthly_sales),
                    'sales_trend': self._calculate_trend(monthly_sales),
                    'seasonality_factor': self._calculate_seasonality(monthly_sales),
                    'growth_rate': self._calculate_growth_rate(monthly_sales),
                    'monthly_sales': monthly_sales
                }
                features_list.append(features)
        
        return features_list
    
    def _calculate_trend(self, sales_data):
        """Calculate trend using linear regression slope"""
        if len(sales_data) < 2:
            return 0
        x = np.arange(len(sales_data))
        slope = np.polyfit(x, sales_data, 1)[0]
        return slope
    
    def _calculate_seasonality(self, sales_data):
        """Simple seasonality factor based on coefficient of variation"""
        if len(sales_data) < 2 or np.mean(sales_data) == 0:
            return 0
        return np.std(sales_data) / np.mean(sales_data)
    
    def _calculate_growth_rate(self, sales_data):
        """Calculate growth rate from first to last period"""
        if len(sales_data) < 2 or sales_data[0] == 0:
            return 0
        return (sales_data[-1] - sales_data[0]) / sales_data[0]
    
    def train_models(self, year):
        """
        Train prediction models using historical data
        """
        if year not in self.training_data:
            print(f"No training data found for {year}")
            return
        
        features_list = self.prepare_features(self.training_data[year])
        
        if not features_list:
            print(f"No valid features extracted for {year}")
            return
        
        # Prepare training data for next year prediction
        X = []
        y_totals = []
        item_codes = []
        
        for feature_dict in features_list:
            # Features for prediction
            feature_vector = [
                feature_dict['avg_monthly_sales'],
                feature_dict['max_monthly_sales'],
                feature_dict['min_monthly_sales'],
                feature_dict['sales_std'],
                feature_dict['sales_trend'],
                feature_dict['seasonality_factor'],
                feature_dict['growth_rate']
            ]
            
            X.append(feature_vector)
            y_totals.append(feature_dict['total_sales'])
            item_codes.append(feature_dict['item_code'])
        
        X = np.array(X)
        y_totals = np.array(y_totals)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train models
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        lr_model = LinearRegression()
        
        rf_model.fit(X_scaled, y_totals)
        lr_model.fit(X_scaled, y_totals)
        
        # Store models and scaler
        self.models[year] = {
            'random_forest': rf_model,
            'linear_regression': lr_model,
            'item_codes': item_codes,
            'features': X,
            'targets': y_totals
        }
        self.scalers[year] = scaler
        
        print(f"Models trained for {year} with {len(item_codes)} parts")
    
    def predict_next_year(self, base_year, prediction_year):
        """
        Predict sales for the next year based on base year data
        """
        if base_year not in self.models:
            print(f"No trained model found for {base_year}")
            return
        
        model_data = self.models[base_year]
        scaler = self.scalers[base_year]
        
        # Make predictions
        X_scaled = scaler.transform(model_data['features'])
        
        rf_predictions = model_data['random_forest'].predict(X_scaled)
        lr_predictions = model_data['linear_regression'].predict(X_scaled)
        
        # Ensemble prediction (average of both models)
        ensemble_predictions = (rf_predictions + lr_predictions) / 2
        
        # Create predictions dataframe
        predictions_df = pd.DataFrame({
            'Item_Code': model_data['item_codes'],
            'RF_Prediction': rf_predictions,
            'LR_Prediction': lr_predictions,
            'Ensemble_Prediction': ensemble_predictions,
            'Historical_Total': model_data['targets']
        })
        
        self.predictions[prediction_year] = predictions_df
        
        print(f"Predictions generated for {prediction_year} based on {base_year} data")
        return predictions_df
    
    def compare_predictions(self, prediction_year, actual_year):
        """
        Compare predictions with actual results
        """
        if prediction_year not in self.predictions:
            print(f"No predictions found for {prediction_year}")
            return
        
        if actual_year not in self.training_data:
            print(f"No actual data found for {actual_year}")
            return
        
        predictions_df = self.predictions[prediction_year]
        actual_features = self.prepare_features(self.training_data[actual_year])
        
        # Create actual results dataframe
        actual_df = pd.DataFrame([
            {
                'Item_Code': f['item_code'],
                'Actual_Total': f['total_sales']
            }
            for f in actual_features
        ])
        
        # Merge predictions with actual results
        comparison_df = predictions_df.merge(actual_df, on='Item_Code', how='inner')
        
        if comparison_df.empty:
            print("No matching items found between predictions and actual data")
            return
        
        # Calculate accuracy metrics
        metrics = {}
        for pred_col in ['RF_Prediction', 'LR_Prediction', 'Ensemble_Prediction']:
            mae = mean_absolute_error(comparison_df['Actual_Total'], comparison_df[pred_col])
            mse = mean_squared_error(comparison_df['Actual_Total'], comparison_df[pred_col])
            r2 = r2_score(comparison_df['Actual_Total'], comparison_df[pred_col])
            
            metrics[pred_col] = {
                'MAE': mae,
                'RMSE': np.sqrt(mse),
                'R²': r2,
                'MAPE': np.mean(np.abs((comparison_df['Actual_Total'] - comparison_df[pred_col]) / 
                                     comparison_df['Actual_Total'])) * 100
            }
        
        return comparison_df, metrics
    
    def visualize_comparison(self, comparison_df, metrics, prediction_year):
        """
        Create visualizations for prediction vs actual comparison
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Prediction vs Actual Comparison - {prediction_year}', fontsize=16)
        
        # Scatter plot: Predicted vs Actual
        ax1 = axes[0, 0]
        ax1.scatter(comparison_df['Actual_Total'], comparison_df['Ensemble_Prediction'], alpha=0.6)
        ax1.plot([comparison_df['Actual_Total'].min(), comparison_df['Actual_Total'].max()], 
                [comparison_df['Actual_Total'].min(), comparison_df['Actual_Total'].max()], 'r--')
        ax1.set_xlabel('Actual Sales')
        ax1.set_ylabel('Predicted Sales')
        ax1.set_title('Predicted vs Actual Sales')
        
        # Error distribution
        ax2 = axes[0, 1]
        errors = comparison_df['Actual_Total'] - comparison_df['Ensemble_Prediction']
        ax2.hist(errors, bins=30, alpha=0.7)
        ax2.set_xlabel('Prediction Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Prediction Error Distribution')
        
        # Top 20 items comparison
        ax3 = axes[1, 0]
        top_20 = comparison_df.nlargest(20, 'Actual_Total')
        x_pos = np.arange(len(top_20))
        ax3.bar(x_pos - 0.2, top_20['Actual_Total'], 0.4, label='Actual', alpha=0.7)
        ax3.bar(x_pos + 0.2, top_20['Ensemble_Prediction'], 0.4, label='Predicted', alpha=0.7)
        ax3.set_xlabel('Top 20 Items')
        ax3.set_ylabel('Sales Volume')
        ax3.set_title('Top 20 Items: Actual vs Predicted')
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
        
        # Metrics comparison
        ax4 = axes[1, 1]
        model_names = list(metrics.keys())
        mae_values = [metrics[model]['MAE'] for model in model_names]
        r2_values = [metrics[model]['R²'] for model in model_names]
        
        x_pos = np.arange(len(model_names))
        ax4_twin = ax4.twinx()
        
        bars1 = ax4.bar(x_pos - 0.2, mae_values, 0.4, label='MAE', alpha=0.7)
        bars2 = ax4_twin.bar(x_pos + 0.2, r2_values, 0.4, label='R²', alpha=0.7, color='orange')
        
        ax4.set_xlabel('Models')
        ax4.set_ylabel('MAE', color='blue')
        ax4_twin.set_ylabel('R²', color='orange')
        ax4.set_title('Model Performance Comparison')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([name.replace('_', ' ') for name in model_names], rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Print metrics summary
        print("\nModel Performance Metrics:")
        print("=" * 50)
        for model, metric in metrics.items():
            print(f"\n{model.replace('_', ' ')}:")
            print(f"  MAE: {metric['MAE']:.2f}")
            print(f"  RMSE: {metric['RMSE']:.2f}")
            print(f"  R²: {metric['R²']:.3f}")
            print(f"  MAPE: {metric['MAPE']:.2f}%")

# Example usage
def main():
    """
    Example usage of the SparePartsPredictor
    """
    predictor = SparePartsPredictor()
    
    # Load data files
    print("Loading data files...")
    # predictor.load_data('spare_parts_2022.xlsx', 2022)  # Your 2022 data
    # predictor.load_data('spare_parts_2023.xlsx', 2023)  # Your 2023 data for comparison
    
    # Train model on 2022 data
    print("Training models...")
    # predictor.train_models(2022)
    
    # Predict 2023 based on 2022
    print("Making predictions...")
    # predictions_2023 = predictor.predict_next_year(2022, 2023)
    
    # Compare predictions with actual 2023 data
    print("Comparing predictions with actual results...")
    # comparison_df, metrics = predictor.compare_predictions(2023, 2023)
    
    # Visualize results
    # predictor.visualize_comparison(comparison_df, metrics, 2023)
    
    print("Analysis complete!")

if __name__ == "__main__":
    # Uncomment the line below to run the example
    # main()
    
    # Instructions for usage:
    print("""
    Usage Instructions:
    
    1. Save your Excel/CSV files with the format shown in your screenshot
    2. Create a SparePartsPredictor instance:
       predictor = SparePartsPredictor()
    
    3. Load your data files:
       predictor.load_data('your_2022_file.xlsx', 2022)
       predictor.load_data('your_2023_file.xlsx', 2023)
    
    4. Train the model:
       predictor.train_models(2022)
    
    5. Make predictions:
       predictions = predictor.predict_next_year(2022, 2023)
    
    6. Compare with actual results:
       comparison, metrics = predictor.compare_predictions(2023, 2023)
    
    7. Visualize results:
       predictor.visualize_comparison(comparison, metrics, 2023)
    """)
