# Bike Sharing Machine Learning Model for D.C.
This dataset comes from bike rentals in the city of Washington D.C. in 2011, 2012. 
---
## Goal:
The goal of our regression model is to train our model to predict the hourly bike rentals for the bike system hourly using information from the data. Explanation of data received can be found at the top of the notebook.

## Models Used:
- RandomForestRegressor, LGBMRegressor

---

## Files Included:
1. Exploration_Viz_Model.ipynb
- This is the notebook that has all of our code in it, including EDA and machine learning models. We tried baseline models of Random Forest to compare our final model to, all provided in the notebook.

2. Script_Streamlit.py
- This is the script that you need to run to see the streamlit website.
- Command to run:

```python
streamlit run Script_Streamlit.py
```
3. PredictionScript.py 
- This script receives the cleaned data and holds the processing and preparation for the model LightLGBM that we decided with the hyperparameters we found were the best. You do not need to run this script, it is called by the other script. Make sure they are saved in the same directory.

4. bike_sharing_modify.csv 
- Cleaned data to used for our model and for the Prediction.py script.
Attaching it just in case, our notebook creates it as the .csv.

5. bike-sharing-hourly.csv
- Original CSV untouched with the data.

## Potential Future Improvements:
- With more data, we could train a better model to predict, given that the data has likely changed from 2011/2012. To provide better estimates, we would need to secure more data.
- Due to computational limitations, we only ran 2 models to include in our final solution. We could easily run more models to see if a more basic model has a better rate of prediction.

