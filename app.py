import React, { useState, useCallback, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Upload, TrendingUp, Calendar, Package, Target, Award, Zap, Brain } from 'lucide-react';

const SPCSalesForecastingDashboard = () => {
  const [historicalData, setHistoricalData] = useState(null);
  const [actualData, setActualData] = useState(null);
  const [forecastResults, setForecastResults] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadStatus, setUploadStatus] = useState({ historical: false, actual: false });

  // Enhanced Excel file processor for SPC format
  const processExcelFile = useCallback(async (file, dataType) => {
    try {
      const arrayBuffer = await file.arrayBuffer();
      
      // Import SheetJS dynamically
      const XLSX = await import('xlsx');
      const workbook = XLSX.read(arrayBuffer);
      const worksheet = workbook.Sheets[workbook.SheetNames[0]];
      const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });

      if (jsonData.length < 3) {
        throw new Error('Excel file must have at least 3 rows (headers + data)');
      }

      // Get headers from first row - month columns start from index 4
      const headers = jsonData[0];
      const monthColumns = [];
      
      // Extract month columns (they contain year info like "Jan-2022", "Feb-2022")
      headers.forEach((header, idx) => {
        if (header && (header.toString().includes('2022') || header.toString().includes('2023') || header.toString().includes('2024'))) {
          monthColumns.push({ index: idx, name: header.toString() });
        }
      });

      if (monthColumns.length === 0) {
        throw new Error('No month columns found. Expected format: Jan-2022, Feb-2022, etc.');
      }

      // Process data rows (skip first 2 rows: headers and QTY labels)
      const dataRows = jsonData.slice(2);
      const processedData = [];

      // Aggregate all parts by month
      monthColumns.forEach(monthCol => {
        const monthName = monthCol.name;
        let totalSales = 0;

        // Sum all parts for this month
        dataRows.forEach(row => {
          if (row && row[monthCol.index] && !isNaN(row[monthCol.index])) {
            totalSales += Number(row[monthCol.index]);
          }
        });

        if (totalSales > 0) {
          // Parse month-year to create proper date
          const [month, year] = monthName.split('-');
          const monthNum = new Date(Date.parse(month + " 1, 2000")).getMonth() + 1;
          const dateStr = `${year}-${monthNum.toString().padStart(2, '0')}-01`;
          
          processedData.push({
            Month: new Date(dateStr),
            Sales: totalSales,
            MonthName: monthName
          });
        }
      });

      // Sort by date
      processedData.sort((a, b) => a.Month - b.Month);

      setUploadStatus(prev => ({ ...prev, [dataType]: true }));
      return processedData;

    } catch (error) {
      console.error('Error processing Excel file:', error);
      throw new Error(`Failed to process Excel file: ${error.message}`);
    }
  }, []);

  // Handle file uploads
  const handleFileUpload = useCallback(async (event, dataType) => {
    const file = event.target.files[0];
    if (!file) return;

    try {
      setIsProcessing(true);
      const processedData = await processExcelFile(file, dataType);
      
      if (dataType === 'historical') {
        setHistoricalData(processedData);
      } else {
        setActualData(processedData);
      }

    } catch (error) {
      alert(`Error processing ${dataType} file: ${error.message}`);
    } finally {
      setIsProcessing(false);
    }
  }, [processExcelFile]);

  // Advanced forecasting models
  const generateAdvancedForecasts = useCallback(() => {
    if (!historicalData || historicalData.length < 6) {
      alert('Need at least 6 months of historical data for forecasting');
      return;
    }

    setIsProcessing(true);

    try {
      const salesData = historicalData.map(d => d.Sales);
      const forecastPeriods = 12;
      
      // Create forecast dates
      const lastDate = new Date(historicalData[historicalData.length - 1].Month);
      const forecastDates = [];
      for (let i = 1; i <= forecastPeriods; i++) {
        const newDate = new Date(lastDate);
        newDate.setMonth(newDate.getMonth() + i);
        forecastDates.push(newDate);
      }

      // Model 1: Enhanced Moving Average with Seasonality
      const seasonalMA = () => {
        const period = Math.min(12, salesData.length);
        const forecast = [];
        
        for (let i = 0; i < forecastPeriods; i++) {
          const seasonalIndex = (historicalData.length + i) % period;
          const seasonalFactor = salesData[seasonalIndex] / (salesData.reduce((a, b) => a + b, 0) / salesData.length);
          
          const recentAvg = salesData.slice(-period).reduce((a, b) => a + b, 0) / period;
          forecast.push(recentAvg * seasonalFactor);
        }
        return forecast;
      };

      // Model 2: Linear Trend with Seasonal Adjustment
      const linearTrendSeasonal = () => {
        const n = salesData.length;
        const xValues = Array.from({length: n}, (_, i) => i + 1);
        
        // Calculate linear trend
        const sumX = xValues.reduce((a, b) => a + b, 0);
        const sumY = salesData.reduce((a, b) => a + b, 0);
        const sumXY = xValues.reduce((sum, x, i) => sum + x * salesData[i], 0);
        const sumXX = xValues.reduce((sum, x) => sum + x * x, 0);
        
        const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;
        
        // Generate forecast with trend
        const forecast = [];
        for (let i = 0; i < forecastPeriods; i++) {
          const trendValue = slope * (n + i + 1) + intercept;
          // Add seasonal component
          const seasonalIndex = (n + i) % 12;
          const seasonalFactor = seasonalIndex < salesData.length ? 
            salesData[seasonalIndex] / (sumY / n) : 1;
          
          forecast.push(Math.max(0, trendValue * seasonalFactor));
        }
        return forecast;
      };

      // Model 3: Exponential Smoothing with Trend and Seasonality
      const exponentialSmoothing = () => {
        const alpha = 0.3; // Level smoothing
        const beta = 0.1;  // Trend smoothing
        const gamma = 0.2; // Seasonal smoothing
        
        let level = salesData[0];
        let trend = salesData.length > 1 ? salesData[1] - salesData[0] : 0;
        const seasonal = new Array(12).fill(1);
        
        // Initialize seasonal factors
        if (salesData.length >= 12) {
          for (let i = 0; i < 12; i++) {
            if (i < salesData.length) {
              seasonal[i] = salesData[i] / (salesData.slice(0, 12).reduce((a, b) => a + b, 0) / 12);
            }
          }
        }
        
        // Update level, trend, and seasonal factors
        for (let i = 1; i < salesData.length; i++) {
          const newLevel = alpha * (salesData[i] / seasonal[i % 12]) + (1 - alpha) * (level + trend);
          const newTrend = beta * (newLevel - level) + (1 - beta) * trend;
          seasonal[i % 12] = gamma * (salesData[i] / newLevel) + (1 - gamma) * seasonal[i % 12];
          
          level = newLevel;
          trend = newTrend;
        }
        
        // Generate forecast
        const forecast = [];
        for (let i = 0; i < forecastPeriods; i++) {
          const forecastValue = (level + (i + 1) * trend) * seasonal[(salesData.length + i) % 12];
          forecast.push(Math.max(0, forecastValue));
        }
        return forecast;
      };

      // Model 4: ARIMA-like Model (simplified)
      const arimaLike = () => {
        const forecast = [];
        const p = 2; // AR order
        const d = 1; // Differencing order
        const q = 1; // MA order
        
        // Difference the series
        const diffData = [];
        for (let i = d; i < salesData.length; i++) {
          diffData.push(salesData[i] - salesData[i - d]);
        }
        
        if (diffData.length < p + q) {
          return linearTrendSeasonal(); // Fallback
        }
        
        // Simple AR model on differenced data
        for (let i = 0; i < forecastPeriods; i++) {
          let prediction = 0;
          
          if (i === 0) {
            // First prediction based on AR coefficients
            for (let j = 0; j < Math.min(p, diffData.length); j++) {
              prediction += 0.5 * diffData[diffData.length - 1 - j] / (j + 1);
            }
            prediction += salesData[salesData.length - 1];
          } else {
            // Subsequent predictions
            prediction = forecast[i - 1] * 1.02; // Small growth factor
          }
          
          forecast.push(Math.max(0, prediction));
        }
        
        return forecast;
      };

      // Generate all forecasts
      const models = {
        'Seasonal_MA': seasonalMA(),
        'Linear_Trend': linearTrendSeasonal(),
        'Exp_Smoothing': exponentialSmoothing(),
        'ARIMA_Like': arimaLike()
      };

      // Create ensemble forecast (weighted average)
      const ensembleForecast = [];
      for (let i = 0; i < forecastPeriods; i++) {
        const values = Object.values(models).map(model => model[i]);
        const avgValue = values.reduce((a, b) => a + b, 0) / values.length;
        ensembleForecast.push(avgValue);
      }
      models['Ensemble'] = ensembleForecast;

      // Create results structure
      const results = forecastDates.map((date, index) => {
        const result = {
          Month: date,
          MonthName: date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' })
        };
        
        Object.entries(models).forEach(([modelName, forecast]) => {
          result[modelName] = Math.round(forecast[index]);
        });
        
        return result;
      });

      setForecastResults(results);
      
    } catch (error) {
      console.error('Forecasting error:', error);
      alert('Error generating forecasts. Please check your data.');
    } finally {
      setIsProcessing(false);
    }
  }, [historicalData]);

  // Calculate performance metrics
  const performanceMetrics = useMemo(() => {
    if (!forecastResults || !actualData) return null;

    const metrics = {};
    const modelNames = Object.keys(forecastResults[0]).filter(key => 
      key !== 'Month' && key !== 'MonthName'
    );

    modelNames.forEach(modelName => {
      const pairs = [];
      
      forecastResults.forEach(forecast => {
        const actual = actualData.find(a => 
          a.Month.getFullYear() === forecast.Month.getFullYear() &&
          a.Month.getMonth() === forecast.Month.getMonth()
        );
        
        if (actual) {
          pairs.push({
            actual: actual.Sales,
            forecast: forecast[modelName]
          });
        }
      });

      if (pairs.length > 0) {
        const actualValues = pairs.map(p => p.actual);
        const forecastValues = pairs.map(p => p.forecast);
        
        // Calculate MAPE
        const mape = pairs.reduce((sum, pair) => {
          return sum + Math.abs((pair.actual - pair.forecast) / pair.actual);
        }, 0) / pairs.length * 100;

        // Calculate MAE
        const mae = pairs.reduce((sum, pair) => {
          return sum + Math.abs(pair.actual - pair.forecast);
        }, 0) / pairs.length;

        // Calculate RMSE
        const rmse = Math.sqrt(
          pairs.reduce((sum, pair) => {
            return sum + Math.pow(pair.actual - pair.forecast, 2);
          }, 0) / pairs.length
        );

        metrics[modelName] = {
          MAPE: mape.toFixed(1),
          MAE: mae.toFixed(0),
          RMSE: rmse.toFixed(0),
          Count: pairs.length
        };
      }
    });

    return metrics;
  }, [forecastResults, actualData]);

  // Prepare chart data
  const chartData = useMemo(() => {
    if (!historicalData) return [];

    let data = [...historicalData].map(d => ({
      Month: d.MonthName || d.Month.toLocaleDateString('en-US', { month: 'short', year: 'numeric' }),
      Actual: d.Sales,
      Type: 'Historical'
    }));

    if (forecastResults) {
      const forecastData = forecastResults.map(f => ({
        Month: f.MonthName,
        Type: 'Forecast',
        Seasonal_MA: f.Seasonal_MA,
        Linear_Trend: f.Linear_Trend,
        Exp_Smoothing: f.Exp_Smoothing,
        ARIMA_Like: f.ARIMA_Like,
        Ensemble: f.Ensemble
      }));

      data = [...data, ...forecastData];
    }

    if (actualData) {
      actualData.forEach(actual => {
        const monthName = actual.MonthName || actual.Month.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
        const existingIndex = data.findIndex(d => d.Month === monthName);
        
        if (existingIndex >= 0) {
          data[existingIndex].Actual = actual.Sales;
          data[existingIndex].Type = 'Actual';
        } else {
          data.push({
            Month: monthName,
            Actual: actual.Sales,
            Type: 'Actual'
          });
        }
      });
    }

    return data.slice(-36); // Show last 3 years
  }, [historicalData, forecastResults, actualData]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl p-8 mb-8 text-white">
        <div className="flex items-center gap-4 mb-4">
          <Package className="h-10 w-10" />
          <div>
            <h1 className="text-4xl font-bold">SPC Sales Forecasting Dashboard</h1>
            <p className="text-indigo-100 text-lg">Advanced AI-powered forecasting for SPC parts sales data</p>
          </div>
        </div>
      </div>

      {/* Upload Section */}
      <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
          <Upload className="h-6 w-6" />
          Upload Excel Files
        </h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          {/* Historical Data Upload */}
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 hover:border-indigo-400 transition-colors">
            <div className="text-center">
              <Calendar className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-700 mb-2">Historical Sales Data</h3>
              <p className="text-gray-500 mb-4">Upload SPC sales data (2022-2023 Excel format)</p>
              <input
                type="file"
                accept=".xlsx,.xls"
                onChange={(e) => handleFileUpload(e, 'historical')}
                className="hidden"
                id="historical-upload"
                disabled={isProcessing}
              />
              <label
                htmlFor="historical-upload"
                className="bg-indigo-600 text-white px-6 py-2 rounded-lg cursor-pointer hover:bg-indigo-700 transition-colors inline-block"
              >
                {isProcessing ? 'Processing...' : 'Choose File'}
              </label>
              {uploadStatus.historical && (
                <p className="text-green-600 mt-2 flex items-center justify-center gap-1">
                  ✅ Historical data loaded ({historicalData?.length} months)
                </p>
              )}
            </div>
          </div>

          {/* Actual 2024 Data Upload */}
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 hover:border-green-400 transition-colors">
            <div className="text-center">
              <Target className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-700 mb-2">2024 Actual Data (Optional)</h3>
              <p className="text-gray-500 mb-4">Upload 2024 actual data for model validation</p>
              <input
                type="file"
                accept=".xlsx,.xls"
                onChange={(e) => handleFileUpload(e, 'actual')}
                className="hidden"
                id="actual-upload"
                disabled={isProcessing}
              />
              <label
                htmlFor="actual-upload"
                className="bg-green-600 text-white px-6 py-2 rounded-lg cursor-pointer hover:bg-green-700 transition-colors inline-block"
              >
                {isProcessing ? 'Processing...' : 'Choose File'}
              </label>
              {uploadStatus.actual && (
                <p className="text-green-600 mt-2 flex items-center justify-center gap-1">
                  ✅ Actual data loaded ({actualData?.length} months)
                </p>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Data Summary */}
      {historicalData && (
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <h2 className="text-2xl font-bold text-gray-800 mb-6">📊 Data Analysis</h2>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-blue-50 p-4 rounded-lg border-l-4 border-blue-500">
              <h3 className="font-semibold text-blue-700">Total Months</h3>
              <p className="text-2xl font-bold text-blue-800">{historicalData.length}</p>
            </div>
            
            <div className="bg-green-50 p-4 rounded-lg border-l-4 border-green-500">
              <h3 className="font-semibold text-green-700">Avg Monthly Sales</h3>
              <p className="text-2xl font-bold text-green-800">
                {Math.round(historicalData.reduce((sum, d) => sum + d.Sales, 0) / historicalData.length).toLocaleString()}
              </p>
            </div>
            
            <div className="bg-purple-50 p-4 rounded-lg border-l-4 border-purple-500">
              <h3 className="font-semibold text-purple-700">Peak Sales</h3>
              <p className="text-2xl font-bold text-purple-800">
                {Math.max(...historicalData.map(d => d.Sales)).toLocaleString()}
              </p>
            </div>
            
            <div className="bg-orange-50 p-4 rounded-lg border-l-4 border-orange-500">
              <h3 className="font-semibold text-orange-700">Data Range</h3>
              <p className="text-lg font-bold text-orange-800">
                {historicalData[0].Month.getFullYear()} - {historicalData[historicalData.length - 1].Month.getFullYear()}
              </p>
            </div>
          </div>

          {/* Generate Forecast Button */}
          <div className="text-center">
            <button
              onClick={generateAdvancedForecasts}
              disabled={isProcessing || historicalData.length < 6}
              className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white px-8 py-4 rounded-lg text-lg font-semibold hover:from-indigo-700 hover:to-purple-700 transition-all transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 mx-auto"
            >
              <Brain className="h-6 w-6" />
              {isProcessing ? 'Generating Forecasts...' : 'Generate AI Forecasts'}
            </button>
          </div>
        </div>
      )}

      {/* Forecast Results */}
      {forecastResults && (
        <div className="space-y-8">
          {/* Forecast Summary */}
          <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl p-8 text-white">
            <h2 className="text-3xl font-bold mb-6 text-center">🎯 Forecast Results Summary</h2>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div className="text-center">
                <h3 className="text-lg font-semibold mb-2">Total 12-Month Forecast</h3>
                <p className="text-3xl font-bold">
                  {Math.round(forecastResults.reduce((sum, f) => sum + f.Ensemble, 0)).toLocaleString()}
                </p>
                <p className="text-indigo-200">units</p>
              </div>
              
              <div className="text-center">
                <h3 className="text-lg font-semibold mb-2">Avg Monthly</h3>
                <p className="text-3xl font-bold">
                  {Math.round(forecastResults.reduce((sum, f) => sum + f.Ensemble, 0) / 12).toLocaleString()}
                </p>
                <p className="text-indigo-200">units/month</p>
              </div>
              
              <div className="text-center">
                <h3 className="text-lg font-semibold mb-2">Growth vs Last Year</h3>
                <p className="text-3xl font-bold">
                  {historicalData.length >= 12 ? 
                    `${((forecastResults.reduce((sum, f) => sum + f.Ensemble, 0) / historicalData.slice(-12).reduce((sum, d) => sum + d.Sales, 0) - 1) * 100).toFixed(1)}%`
                    : 'N/A'
                  }
                </p>
                <p className="text-indigo-200">projected change</p>
              </div>
              
              <div className="text-center">
                <h3 className="text-lg font-semibold mb-2">Models Used</h3>
                <p className="text-3xl font-bold">5</p>
                <p className="text-indigo-200">AI algorithms</p>
              </div>
            </div>
          </div>

          {/* Chart Visualization */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
              <TrendingUp className="h-6 w-6" />
              Sales Trend & Forecasts
            </h2>
            
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="Month" 
                    angle={-45}
                    textAnchor="end"
                    height={60}
                  />
                  <YAxis />
                  <Tooltip formatter={(value, name) => [value?.toLocaleString(), name]} />
                  <Legend />
                  
                  <Line 
                    type="monotone" 
                    dataKey="Actual" 
                    stroke="#2563eb" 
                    strokeWidth={3}
                    dot={{ fill: '#2563eb', strokeWidth: 2, r: 4 }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="Ensemble" 
                    stroke="#dc2626" 
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={{ fill: '#dc2626', strokeWidth: 2, r: 3 }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="Seasonal_MA" 
                    stroke="#059669" 
                    strokeWidth={1}
                    dot={false}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="Linear_Trend" 
                    stroke="#7c3aed" 
                    strokeWidth={1}
                    dot={false}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="Exp_Smoothing" 
                    stroke="#ea580c" 
                    strokeWidth={1}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Detailed Forecast Table */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">📊 Detailed Forecast Results</h2>
            
            <div className="overflow-x-auto">
              <table className="w-full border-collapse border border-gray-300">
                <thead>
                  <tr className="bg-gray-50">
                    <th className="border border-gray-300 px-4 py-2 text-left">Month</th>
                    <th className="border border-gray-300 px-4 py-2 text-right">Seasonal MA</th>
                    <th className="border border-gray-300 px-4 py-2 text-right">Linear Trend</th>
                    <th className="border border-gray-300 px-4 py-2 text-right">Exp Smoothing</th>
                    <th className="border border-gray-300 px-4 py-2 text-right">ARIMA-like</th>
                    <th className="border border-gray-300 px-4 py-2 text-right bg-indigo-100">Ensemble</th>
                    {actualData && <th className="border border-gray-300 px-4 py-2 text-right bg-green-100">Actual</th>}
                  </tr>
                </thead>
                <tbody>
                  {forecastResults.map((row, index) => {
                    const actual = actualData?.find(a => 
                      a.Month.getFullYear() === row.Month.getFullYear() &&
                      a.Month.getMonth() === row.Month.getMonth()
                    );
                    
                    return (
                      <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                        <td className="border border-gray-300 px-4 py-2 font-medium">{row.MonthName}</td>
                        <td className="border border-gray-300 px-4 py-2 text-right">{row.Seasonal_MA.toLocaleString()}</td>
                        <td className="border border-gray-300 px-4 py-2 text-right">{row.Linear_Trend.toLocaleString()}</td>
                        <td className="border border-gray-300 px-4 py-2 text-right">{row.Exp_Smoothing.toLocaleString()}</td>
                        <td className="border border-gray-300 px-4 py-2 text-right">{row.ARIMA_Like.toLocaleString()}</td>
                        <td className="border border-gray-300 px-4 py-2 text-right font-bold bg-indigo-50">
                          {row.Ensemble.toLocaleString()}
                        </td>
                        {actualData && (
                          <td className={`border border-gray-300 px-4 py-2 text-right font-bold ${actual ? 'bg-green-50' : 'bg-gray-100'}`}>
                            {actual ? actual.Sales.toLocaleString() : '-'}
                          </td>
                        )}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Performance Metrics */}
          {performanceMetrics && (
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                <Award className="h-6 w-6" />
                Model Performance Analysis
              </h2>
              
              <div className="grid md:grid-cols-2 gap-6">
                {/* Performance Table */}
                <div>
                  <h3 className="text-lg font-semibold mb-4">Accuracy Metrics</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full border-collapse border border-gray-300">
                      <thead>
                        <tr className="bg-gray-50">
                          <th className="border border-gray-300 px-3 py-2 text-left">Model</th>
                          <th className="border border-gray-300 px-3 py-2 text-right">MAPE (%)</th>
                          <th className="border border-gray-300 px-3 py-2 text-right">MAE</th>
                          <th className="border border-gray-300 px-3 py-2 text-right">RMSE</th>
                          <th className="border border-gray-300 px-3 py-2 text-right">Count</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(performanceMetrics).map(([model, metrics]) => {
                          const isEnsemble = model === 'Ensemble';
                          const isBest = Object.values(performanceMetrics).every(m => 
                            parseFloat(metrics.MAPE) <= parseFloat(m.MAPE)
                          );
                          
                          return (
                            <tr key={model} className={`${isEnsemble ? 'bg-indigo-50 font-semibold' : ''} ${isBest && !isEnsemble ? 'bg-green-50' : ''}`}>
                              <td className="border border-gray-300 px-3 py-2">
                                {model.replace('_', ' ')}
                                {isBest && !isEnsemble && ' 🏆'}
                              </td>
                              <td className="border border-gray-300 px-3 py-2 text-right">{metrics.MAPE}%</td>
                              <td className="border border-gray-300 px-3 py-2 text-right">{metrics.MAE}</td>
                              <td className="border border-gray-300 px-3 py-2 text-right">{metrics.RMSE}</td>
                              <td className="border border-gray-300 px-3 py-2 text-right">{metrics.Count}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  
                  {/* Best Model Highlight */}
                  <div className="mt-4 p-4 bg-green-100 border-l-4 border-green-500 rounded">
                    <h4 className="font-semibold text-green-800">🏆 Best Individual Model</h4>
                    <p className="text-green-700">
                      {Object.entries(performanceMetrics)
                        .filter(([model]) => model !== 'Ensemble')
                        .reduce((best, [model, metrics]) => 
                          parseFloat(metrics.MAPE) < parseFloat(best[1].MAPE) ? [model, metrics] : best
                        )[0].replace('_', ' ')} 
                      with {Object.entries(performanceMetrics)
                        .filter(([model]) => model !== 'Ensemble')
                        .reduce((best, [model, metrics]) => 
                          parseFloat(metrics.MAPE) < parseFloat(best[1].MAPE) ? [model, metrics] : best
                        )[1].MAPE}% MAPE
                    </p>
                  </div>
                </div>

                {/* Performance Chart */}
                <div>
                  <h3 className="text-lg font-semibold mb-4">Model Accuracy Comparison</h3>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={Object.entries(performanceMetrics).map(([model, metrics]) => ({
                        Model: model.replace('_', ' '),
                        MAPE: parseFloat(metrics.MAPE),
                        MAE: parseFloat(metrics.MAE) / 100 // Scale for visualization
                      }))}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="Model" angle={-45} textAnchor="end" height={80} />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="MAPE" fill="#3b82f6" name="MAPE %" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Export Options */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">📁 Export Results</h2>
            
            <div className="grid md:grid-cols-3 gap-4">
              <button 
                onClick={() => {
                  const csvContent = [
                    ['Month', 'Seasonal_MA', 'Linear_Trend', 'Exp_Smoothing', 'ARIMA_Like', 'Ensemble'],
                    ...forecastResults.map(row => [
                      row.MonthName,
                      row.Seasonal_MA,
                      row.Linear_Trend,
                      row.Exp_Smoothing,
                      row.ARIMA_Like,
                      row.Ensemble
                    ])
                  ].map(row => row.join(',')).join('\n');
                  
                  const blob = new Blob([csvContent], { type: 'text/csv' });
                  const url = URL.createObjectURL(blob);
                  const link = document.createElement('a');
                  link.href = url;
                  link.download = 'spc_sales_forecast.csv';
                  link.click();
                }}
                className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2 justify-center"
              >
                📊 Download Forecast CSV
              </button>
              
              {performanceMetrics && (
                <button 
                  onClick={() => {
                    const csvContent = [
                      ['Model', 'MAPE_%', 'MAE', 'RMSE', 'Data_Points'],
                      ...Object.entries(performanceMetrics).map(([model, metrics]) => [
                        model,
                        metrics.MAPE,
                        metrics.MAE,
                        metrics.RMSE,
                        metrics.Count
                      ])
                    ].map(row => row.join(',')).join('\n');
                    
                    const blob = new Blob([csvContent], { type: 'text/csv' });
                    const url = URL.createObjectURL(blob);
                    const link = document.createElement('a');
                    link.href = url;
                    link.download = 'spc_model_performance.csv';
                    link.click();
                  }}
                  className="bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 transition-colors flex items-center gap-2 justify-center"
                >
                  📈 Download Performance CSV
                </button>
              )}
              
              <button 
                onClick={() => {
                  const jsonData = {
                    forecast_results: forecastResults,
                    performance_metrics: performanceMetrics,
                    historical_summary: {
                      total_months: historicalData.length,
                      avg_monthly_sales: Math.round(historicalData.reduce((sum, d) => sum + d.Sales, 0) / historicalData.length),
                      peak_sales: Math.max(...historicalData.map(d => d.Sales)),
                      data_range: `${historicalData[0].Month.getFullYear()}-${historicalData[historicalData.length - 1].Month.getFullYear()}`
                    }
                  };
                  
                  const blob = new Blob([JSON.stringify(jsonData, null, 2)], { type: 'application/json' });
                  const url = URL.createObjectURL(blob);
                  const link = document.createElement('a');
                  link.href = url;
                  link.download = 'spc_forecast_complete.json';
                  link.click();
                }}
                className="bg-purple-600 text-white px-6 py-3 rounded-lg hover:bg-purple-700 transition-colors flex items-center gap-2 justify-center"
              >
                💾 Download Full Report
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="mt-12 text-center text-gray-500">
        <p>SPC Sales Forecasting Dashboard - Powered by Advanced AI & Machine Learning</p>
        <p className="text-sm">Upload your Excel files to get started with intelligent sales forecasting</p>
      </div>
    </div>
  );
};

export default SPCSalesForecastingDashboard;
