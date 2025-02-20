# FiservAI Python Package

This README provides instructions on how to set up and run the app10.py Streamlit application.

## Installation
`pip install streamlit python-dotenv time asyncio regex python-docx DateTime futures threaded`

Update the following variables in the `.ENV` file with what is provided from the Developer Studio workspace
- **API_KEY** =  API Key provided from Developer Studio
- **API_SECRET** = API Secret provided from Developer Studio
- **BASE_URL** = Host URL provided from Developer Studio

- Run the package installation script: `pip3 --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host github.com --trusted-host objects.githubusercontent.com install -r requirements.txt`

- Note: ensure having constants.py in same directory.

## Running the demo
- Run the demo (Windows OS): ` python -m streamlit run.\app.py`
