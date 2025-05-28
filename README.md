# Environment Impact Analyzer

## Overview

A Streamlit web application for analyzing environmental data and assessing city sustainability. The app analyzes air quality, climate patterns, soil characteristics, and water resources to provide insights for informed decision-making.

## Tech Stack

This project is built using the following libraries and frameworks:

*   `streamlit`: For creating the interactive web application interface.
*   `pandas`: For data manipulation and analysis.
*   `numpy`: For numerical operations, particularly in data generation and calculations.
*   `plotly`: For generating interactive data visualizations (line charts, bar charts, pie charts, gauges).
*   `matplotlib`: Although imported, Plotly is primarily used for visualizations in the provided code.
*   `google-generativeai`: For enabling AI features using the Gemini API.

## Setup and Running

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd environment-impact-analyzer
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv .venv
    # On Windows
    .\.venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: AI features require `google-generativeai`. For production, consider using `python-dotenv` or Streamlit secrets for secure API key management.)*

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    *(Replace `app.py` with the actual name of the Python file containing the code)*

## Data

The application generates **sample Pandas DataFrames** for climate, air, soil, and water data. This sample data represents 30 days of simulated environmental metrics for various cities. **Visualizations and analyses are based on this sample data**, providing a functional demonstration of the application's capabilities.

## AI Integration

The application includes optional **AI-powered features** using the `google-generativeai` library and the Gemini API. **While the environmental data used is sample data, the AI model itself is real** and can provide insights and recommendations based on the structure and type of data provided to it, assuming the library is installed and the API key is configured.

## Requirements

Create a `requirements.txt` file with the following dependencies:

```
streamlit
pandas
numpy
plotly
matplotlib
google-generativeai
Pillow
python-dotenv
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
