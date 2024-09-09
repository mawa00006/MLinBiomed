# MLinBiomed

# TODO's:

- modify requirements.txt in the end for upload of assignment


# Run app with

```bash
conda activate streamlit_env
streamlit run main_app.py 
``` 


# Notes Elisa

- stroke = Schlaganfall
- for tabular data preprocessing: append all preprocessing steps to a list and if list not empty join list to a user message about which preprocessing steps have been undertaken
- if e.g. essentially more male/ female in dataset throw warning (bias)
- let user choose imputation (do you want to impute zeros? If yes in which of these categories?)
- do you want to impute by mean or median (median recommended)