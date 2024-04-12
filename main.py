import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sales_data_df = pd.read_csv("sales_data.csv")

# sales_data_df.loc[5:10, 'UnitsSold'] = np.nan
# sales_data_df.loc[15:20, 'Revenue'] = np.nan

# Before Fill
# print(sales_data_df)

# Fill missing values in the UnitsSold column with the mean of the column
# sales_data_df['UnitsSold'].fillna(sales_data_df['UnitsSold'].mean(), inplace=True)
# sales_data_df['Revenue'].fillna(sales_data_df['Revenue'].mean(), inplace=True)

# After Fill
# print(sales_data_df.head())
# print(sales_data_df)

# Step 1: Calculate basic statistics
total_sales = sales_data_df['Revenue'].sum()
average_price = sales_data_df['Revenue'].mean()
total_quanitiy_sold = sales_data_df['UnitsSold'].sum()

print(f'Total Sales: ${total_sales}')
print(f"Average Price: ${average_price:.2f}")
print(f'Total Units Sold: {total_quanitiy_sold}')


# Step 2: Explore sales trends over time
sales_data_df['Date'] = pd.to_datetime(sales_data_df['Date'])  # Convert 'Date' column to datetime
sales_data_df.set_index('Date', inplace=True)  # Set 'Date' column as index

# Group by month and calculate total revenue for each month
monthly_sales = sales_data_df.resample('M').sum()

# Plotting sales trends over time
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales.index, monthly_sales['Revenue'], marker='o')
plt.title('Monthly Sales Trends')
plt.xlabel('Month')
plt.ylabel('Total Revenue')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 3: Analyze the distribution of sales by product category
sales_by_product = sales_data_df.groupby('Product')['Revenue'].sum()

# Plotting distribution of sales by product category
plt.figure(figsize=(8, 6))
sales_by_product.plot(kind='bar')
plt.title('Sales by Product Category')
plt.xlabel('Product')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
'''
Objectives
1. Analyze sales data to gain insights into trends and patterns
2. Forecaast future sales based on historical data
3. Identify correlations between different variables in the dataset
4. Visualize key metrics using plots and charts

Tasks
1. Load the sales data into a Pandas DataFrame
2. Explore the dataset to understand its structure and contents.
3. Preprocess the data by handling missing values, converting data types, and cleaning up the dataset.
4. Perform descriptive statistics to summarize the main characteristics of the data.
5. Visualize sales trends over time using line plots or time series plots.
6. Use statistical methods to identify correlations between sales and other variables.
7. Implement a forecasting model (such as ARIMA or Prophet) to predict future sales.
8. Evaluate the accuracy of the forecasting model using appropriate metrics.
9. Create visualizations to present the results of the analysis and forecasting.
'''