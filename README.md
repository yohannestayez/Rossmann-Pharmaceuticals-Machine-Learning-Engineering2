# Rossmann Pharmaceuticals 

# Sales Forecasting Project

## Overview
This project aims to forecast sales for Rossmann Pharmaceuticals across multiple store locations using historical sales data. The goal is to provide accurate sales predictions six weeks ahead, helping the finance team make informed decisions.

## Table of Contents
- [Business Need](#business-need)
- [Data Description](#data-description)
- [Methodologies](#methodologies)
- [Key Findings](#key-findings)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Business Need
The finance team at Rossmann Pharmaceuticals relies on historical data and personal judgment to forecast sales. This project utilizes machine learning to predict sales, incorporating factors such as promotions, competition, holidays, and seasonal trends.

## Data Description
The dataset contains historical sales data, including:
- Sales figures
- Customer counts
- Promotion details
- Competitor distances
- Store operational status (open/closed)
- Holiday information

## Methodologies
1. **Data Preprocessing**: Cleaned and transformed the data for analysis, handling missing values and converting data types as necessary.
2. **Exploratory Data Analysis (EDA)**: Conducted EDA to identify patterns, trends, and relationships in the data, visualizing findings through graphs and heatmaps.


## Key Findings
- **Sales Trend by Day of the Week**: Sales peak on Mondays and Sundays, with a noticeable dip midweek.
- **Correlation between Sales and Customers**: A strong positive correlation exists between customer counts and sales.
- **Sales and Competitors**: New competitors do not drastically impact sales; distance to competitors shows no clear relationship with sales.

## Installation
To set up this project locally, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/sales-forecasting.git
   ```
2. Navigate into the project directory:
   ```
   cd sales-forecasting
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have suggestions or improvements.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


