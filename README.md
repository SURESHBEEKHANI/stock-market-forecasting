# Stock Market Forecasting

This project is a [Streamlit](https://streamlit.io) web application that forecasts stock prices using ARIMA models. It downloads historical stock data from Yahoo Finance and lets the user forecast selected stock attributes (e.g., Close, High, Low, Open).

## Features

- Fetch historical stock data for a specified ticker and date range.
- Forecast future stock features using an ARIMA model.
- Visualize both historical data and forecasted results.
- Select multiple features to forecast (e.g., Close, High, Low, Open).

## Requirements

- Python 3.8+
- [Streamlit](https://docs.streamlit.io/library/get-started)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [yfinance](https://pypi.org/project/yfinance/)
- [pmdarima](http://alkaline-ml.com/pmdarima/)
- [matplotlib](https://matplotlib.org/)

You can install the required Python packages using:

```bash
pip install streamlit pandas numpy yfinance pmdarima matplotlib
```

## Usage

1. Open a terminal (Command Prompt or PowerShell) on your Windows machine.
2. Navigate to the project folder:

    ```bash
    cd /d f:\stock market forecasting
    ```

3. Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

4. The application will open in your default web browser.

## File Structure

- **app.py**: The main application file.
- **readme.md**: This file with instructions and project information.

## Customization

- You can adjust the forecast days using the slider in the sidebar.
- Change the default stock ticker by modifying the value in the `st.sidebar.text_input` element.
- Modify the custom CSS in the `st.markdown` block to further customize the appearance of the app.

## Troubleshooting

- Ensure that the start date is not set later than the end date.
- If no data is returned, verify that the ticker symbol and date range are correct.
- For any errors during forecast generation, check the console output for details.

## License

This project is licensed under the MIT License.
