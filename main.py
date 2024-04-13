import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings

# Filter out specific warnings
warnings.filterwarnings("ignore", message="A date index has been provided, but it has no associated frequency "
                                          "information.*")
warnings.filterwarnings("ignore", message="A date index has been provided, but it is not monotonic.*")
warnings.filterwarnings("ignore", message="No supported index is available.*")

# Function to load sales data
def load_sales_data(file_path):
    try:
        sales_data_df = pd.read_csv(file_path)
        return sales_data_df
    except FileNotFoundError:
        print("Error: File not found.")
        return None

# Function to preprocess data
def preprocess_data(data):
    # Drop duplicates
    data.drop_duplicates(inplace=True)
    # Handle missing values
    data.fillna(0, inplace=True)
    return data

# Function to calculate basic statistics
def calculate_basic_statistics(data):
    total_sales = data['Revenue'].sum()
    average_price = data['Revenue'].mean()
    total_quantity_sold = data['UnitsSold'].sum()
    return total_sales, average_price, total_quantity_sold

# Function to explore sales trends over time
def explore_sales_trends(data):
    # Convert 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    # Set 'Date' column as index
    data.set_index('Date', inplace=True)
    # Resample data by month
    monthly_sales = data.resample('M').sum()
    # Plot sales trends over time
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_sales.index, monthly_sales['Revenue'], marker='o')
    plt.title('Monthly Sales Trends')
    plt.xlabel('Month')
    plt.ylabel('Total Revenue')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to visualize distribution of sales by product category
def visualize_sales_by_product(data):
    # Group data by product and calculate total revenue
    sales_by_product = data.groupby('Product')['Revenue'].sum()
    # Plot distribution of sales by product category
    plt.figure(figsize=(8, 6))
    sales_by_product.plot(kind='bar')
    plt.title('Sales by Product Category')
    plt.xlabel('Product')
    plt.ylabel('Total Revenue')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

# Function to forecast future sales
def forecast_sales(data):
    # Fit ARIMA model
    model = ARIMA(data['Revenue'], order=(5, 1, 0))
    model_fit = model.fit()
    # Make forecast
    forecast = model_fit.forecast(steps=12)
    return forecast

# Function to evaluate forecasting model
def evaluate_forecast(actual, forecast):
    mse = mean_squared_error(actual, forecast)
    return mse

# Function to present results
def present_results(total_sales, average_price, total_quantity_sold, forecast, mse):
    print(f'Total Sales: ${total_sales}')
    print(f'Average Price: ${average_price:.2f}')
    print(f'Total Quantity Sold: {total_quantity_sold}')
    print(f'Forecasted Sales: {forecast}')
    print(f'Mean Squared Error: {mse}')

# Main function
def main():
    # Step 1: Load sales data
    sales_data = load_sales_data('sales_data.csv')
    if sales_data is None:
        return

    # Step 2: Preprocess the data
    sales_data = preprocess_data(sales_data)

    # Step 3: Calculate basic statistics
    total_sales, average_price, total_quantity_sold = calculate_basic_statistics(sales_data)

    # Step 4: Explore sales trends over time
    explore_sales_trends(sales_data)

    # Step 5: Visualize distribution of sales by product category
    visualize_sales_by_product(sales_data)

    # Step 6: Forecast future sales
    forecast = forecast_sales(sales_data)

    # Step 7: Evaluate the forecasting model
    actual_sales = sales_data['Revenue'][-12:].values
    mse = evaluate_forecast(actual_sales, forecast)

    # Step 8: Present result
    present_results(total_sales, average_price, total_quantity_sold, forecast, mse)

# Entry point of the program
if __name__ == "__main__":
    main()
