import joblib

def PREDICT(data):
    clf = joblib.load("output_models/KNN_model.sav")
    return clf.predict(data)