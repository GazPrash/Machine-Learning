import random
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def dist_rating(rating:str):
    if rating == "Irrelevant":
        return 0
    elif rating == "Negative":
        return 1
    elif rating == "Neutral":
        return 2
    elif rating == "Positive":
        return 3


data = pd.read_csv("data/twitter_training.csv")
data.columns = ["TweetId", "Entity", "Rating", "Text"]
data.dropna(inplace=True)

data.drop_duplicates(subset=["Text"], inplace= True)
data.drop(["TweetId"], axis=1, inplace=True)

entity_bin = (pd.get_dummies(data["Entity"], drop_first=True))
data = pd.concat([data, entity_bin], axis=1)
data.drop(["Entity"], axis=1, inplace=True)
data["Rating"] = data["Rating"].apply(lambda x : dist_rating(x))

xtrain, xtest, ytrain, ytest = train_test_split(data["Text"].values, data["Rating"].values)
vectorizer = CountVectorizer()
corpus = vectorizer.fit_transform(xtrain).toarray()
corpus_test = vectorizer.transform(xtest).toarray()

def train_model():
    model = MultinomialNB()
    model.fit(corpus, ytrain)
    acc = model.score(corpus_test, ytest)
    joblib.dump(model, f"data/twitter_sentnel_model_{acc}acc.pickle")
    joblib.dump(vectorizer, f"data/witter_sentinal_vectorizer_{acc}.pickle")
    return model


# model = train_model()
model:MultinomialNB = joblib.load("data/twitter_sentnel_model_0.7570367812122258acc.pickle")
vectz = joblib.load("data/witter_sentinal_vectorizer_0.7570367812122258.pickle")
validation_data = pd.read_csv("data/twitter_validation.csv")
validation_data.columns = ["TweetId", "Entity", "Rating", "Text"]

# print(validation_data.iloc[212]["Text"])
# text = "Plague of Corruption is #1 on Amazon and # 3 on The NY Times bestseller list"
text1 = str(validation_data.iloc[random.randint(0, 998)]["Text"])

text = """ratio"""

vect_text = vectz.transform([text]).toarray()
print(model.predict(vect_text))
print(text)
