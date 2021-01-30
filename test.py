import pickle

tfidf = pickle.load(open("tfidf.pickle", "rb"))
model = pickle.load(open("model.pickle", "rb"))
id_to_category = pickle.load(open("id_to_category", "rb"))

my_file = open("sample.txt","r")
content = my_file.read()
texts = content.split(";")
my_file.close()
text_features = tfidf.transform(texts)
predictions = model.predict(text_features)
for text, predicted in zip(texts, predictions):
    print('"{}"'.format(text))
    print("  - Predicted as: '{}'".format(id_to_category[predicted]))
    print("")
