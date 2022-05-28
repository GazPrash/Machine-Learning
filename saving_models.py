# Saving models, for instace in a pickle file can be extremly useful
# because it eliminates the need of training your data everytime you need
# to use it to predict stuff...

import pickle
import pandas as pd
from sklearn import linear_model

home_prices = pd.read_csv(
                "data/homeprices.csv"
            )
            
model = linear_model.LinearRegression()
model.fit(home_prices[["area"]].values, home_prices.price.values)


# Saving the model object in a pickle file
# with open("data/model1.pickle", "wb") as f:
#     pickle.dump(model, f)

# Retrieving the model obj
with open("data/model1.pickle", "rb") as file:
    model = pickle.load(file)
    
a = model.predict([[2310]])
print(a)

# Using joblib instead of picke can be more effective especially when your model
# has been trained on data containing a large quantity of numpy arrays.
# Further it also eliminates the need of using a context manager.