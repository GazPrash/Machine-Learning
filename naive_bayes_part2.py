import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


spam_data = pd.read_csv(
    "https://raw.githubusercontent.com/codebasics/py/master/ML/14_naive_bayes/spam.csv"
)

# Actual Trasnformed working data 
data = spam_data.copy(deep= True)
data["Category"] = data["Category"].apply(lambda x : 1 if x == "spam" else 0)
data.rename({"Category" : "Spam"}, axis = 1, inplace=True)

# print(data)
xtrain, xtest, ytrain, ytest = train_test_split(data["Message"].values, data["Spam"].values)

# To analyze or train our model with the above dataset, the message column needs to be converted
# into some relatable numerical form -> So we use Count Vectorization for that
# more info : CountVectorization.png

vectz = CountVectorizer()

training = vectz.fit_transform(xtrain)
corpus_matrix_train = training.toarray()
corpus_matrix_test = vectz.transform(xtest).toarray()

model = MultinomialNB()
model.fit(corpus_matrix_train, ytrain)
acc = model.score(corpus_matrix_test, ytest)

def predict_spam(sample:list):
    tranformed = vectz.transform(sample).toarray()
    if model.predict(tranformed) : print("Yes this is a spam")
    else : print("This ain't no spam")

# predict_spam(["""URGENT Your grandson was arrested last night in Mexico. Need bail money immediately Western Union Wire $9,500
# http://somebullshitlinklol.com"""])

# Reducing code using Pipeline

pline = Pipeline([
    ("vectz", CountVectorizer()),
    ("nbayes", MultinomialNB())
])

pline.fit(xtrain, ytrain)
acc2 = pline.score(xtest, ytest)
# acc == acc2, because using pipeline is the same as using vectz transform everytime individually

print(pline.predict(xtest[23:45]))