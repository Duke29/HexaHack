import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm 
import pickle
spam = pd.read_csv('spamv4.csv')
z_train, z_test, y_train, y_test = train_test_split(spam["text"], spam["type"], test_size=0.12)
cv = CountVectorizer()
features_train = cv.fit_transform(z_train)
with open('cv.pkl','wb') as f:
    pickle.dump(cv,f)
model = svm.SVC()
model.fit(features_train, y_train)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
features_test = cv.transform(z_test)
accuracy = model.score(features_test, y_test)
print("Accuracy: {}".format(accuracy))
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

