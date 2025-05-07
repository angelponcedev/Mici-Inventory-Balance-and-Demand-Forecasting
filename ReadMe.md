# MICI - Demand Forecasting and Inventory Balancing System

A desktop application designed for demand analysis and inventory balancing, developed for MICRON SEMICONDUCTORS in collaboration with ICI for the Talent Land Hackathon 2025. It allows users to load historical data, process it into a weekly format, select eligible products, generate demand forecasts using different models, visualize trends, calculate inventory balance needs, and export results.

## Key Features

*   **Data Loading**: Supports loading historical demand data and configuration parameters from CSV and XLSX files.
*   **Data Processing**: Cleans and transforms historical data into a weekly format. Filters products based on data availability for reliable forecasting.
*   **Interactive Product Selection**: Provides a user-friendly interface to select specific products for analysis and forecasting from the processed dataset.
*   **Demand Forecasting**: Generates weekly demand forecasts for a user-defined horizon using selectable models:
    *   XGBoost
    *   Custom Exponential Smoothing (Model Implemented, Not GUI ready yet)
*   **Boundary Condition Forecasting**: Predicts future boundary conditions (e.g., production capacity) based on historical patterns and writes them back to the input Excel file.
*   **Graphical Visualization**: Displays historical weekly demand alongside the generated forecast using Matplotlib charts integrated directly into the user interface.
*   **Inventory Balance Calculation**: Calculates net demand requirements for selected products based on the forecast, user-provided initial stock, planned receipts, and safety stock levels.
*   **Power BI Export**: Exports historical and forecast data to a CSV file structured for easy use with Power BI, and attempts to automatically open a provided PBIX template file.
*   **Linear Programming Modeling (Planned)**: Future development aims to incorporate linear programming to optimize resource allocation (e.g., man-hours, manufacturing costs) based on production constraints.
*   **Master Production Plan Generation (Planned)**: Future development aims to generate an optimized Master Production Schedule (MPS) based on forecasts and resource optimization results.

## Technologies Used

*   **Language**: Python 3
*   **Graphical User Interface (GUI)**: Tkinter
*   **Data Handling & Processing**: Pandas, NumPy
*   **Visualization**: Matplotlib
*   **Excel Interaction**: openpyxl
*   **Forecasting Models**: XGBoost, Custom Implementations (e.g., Exponential Smoothing)
*   **Linear Programming**: SciPy

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/angelponcedev/Mici-Inventory-Balance-and-Demand-Forecasting.git
    cd Talentland-2025
    ```
2.  **Create a virtual environment**:
    *   The environment name should ideally be `entorno` or `mici` for consistency if collaborating.
    ```bash
    python -m venv entorno
    ```
3.  **Activate the virtual environment**:
    *   Windows:
        ```bash
        .\entorno\Scripts\activate
        ```
    *   macOS/Linux:
        ```bash
        source entorno/bin/activate
        ```
4.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the application**:
    ```bash
    python main.py
    ```
2.  **Load Data**:
    *   Click `Load Dataset` and select your XLSX or CSV file containing historical demand and potentially sheets for 'Weekly Demand Ratio' and 'Boundary Conditions'.
3.  **Process Dataset**:
    *   Click `Process Dataset`. This cleans the data, converts it to a weekly frequency, and identifies eligible products.
4.  **Select Products**:
    *   A pop-up window will appear. Select the products you want to forecast using the checkboxes. Click `Confirm Selection`.
5.  **Generate Forecast**:
    *   Click `Make Forecast` to run XGBoosting modeling for the data.
    *   Select the desired forecast horizon (number of weeks) in the pop-up window.
    *   The forecast will be generated, boundary conditions (if applicable) will be updated in the source Excel, and the chart will display the results for the first selected product.
6.  **Visualize Data**:
    *   The main table shows an overview of the selected products.
    *   Click on different products in the table to update the demand chart.
7.  **Export to Power BI**:
    *   select "Export to Power BI".
    *   Click `Export Data & Open PBIX`. This saves a `predicciones_forecast.csv` file and attempts to open `plantilla_forecast.pbix`.
8.  **Send E-Mail**:
    *   In the "Additional Options & Actions" panel, select "Export to Power BI".
    *   Click `Export Data & Open PBIX`. This saves a `predicciones_forecast.csv` file and attempts to open `plantilla_forecast.pbix`.

## Project Structure

```
Talentland-2025/
├── main.py                 # Main application script, GUI logic
├── dataset/
│   └── dataset.py          # Data loading and processing functions (processDataset)
├── forecast/
│   ├── forecast.py         # Custom Exponential Smoothing forecast logic (Not UI Implemented)
│   └── Xgboost.py          # XGBoost forecast logic
├── boundaryPredictions/
│   └── boundaryFactory.py  # Boundary condition forecasting logic
├── lpm/
│   └── lpm.py             # Linear Programming Model (makeProductionPlan)
├── Assets/
├── plantilla_forecast.pbix # Example Power BI template (optional)
├── requirements.txt        # Python package dependencies
└── README.md               # This file
```

## Limitations and Considerations

*   Input files (CSV/XLSX) should ideally be `UTF-8` encoded. CSV files should use a comma (`,`) as the delimiter.
*   The application expects certain column names in the primary data sheet (e.g., `Product ID`, `Date` or equivalent time period, `Order_Demand` or equivalent demand metric). Refer to `dataset/dataset.py` for specifics if needed.
*   The 'Weekly Demand Ratio' and 'Boundary Conditions' sheets are expected in the XLSX file for full functionality (defaults/warnings are used otherwise).
*   Product eligibility for forecasting is determined during the `Process Dataset` step, typically requiring sufficient historical data points for meaningful weekly conversion and forecasting.
*   Boundary condition updates require the source file to be an XLSX file and write permissions.

## License

Distributed under the MIT License towards Micron SemiConductors. See `LICENSE` file for more details.

---

**Note**: This project was developed for the Hackathon 2025 event, sponsored by Micron Semiconductors. Use of the contents of this repository is strictly subject to the permission of Micron SemiConductors.
