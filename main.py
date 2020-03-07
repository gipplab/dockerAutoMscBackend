import uvicorn
from fastapi import FastAPI
import pickle
import bz2

with open('models/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
# with open('models/classifier.pkl', 'rb') as f:
#     classifier = pickle.load(f)
#     sfile = bz2.BZ2File('models/classifier.pkl.bz2', 'w')
#     pickle.dump(classifier, sfile)
#     sfile.close()
with bz2.open('models/classifier.pkl.bz2', 'rb') as f:
    classifier = pickle.load(f)

app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/classify/")
async def read_item(title: str, text: str, mscs_from_references: str = ""):
    vector = encoder.transform([title + text + mscs_from_references])
    dist = classifier.predict_proba(vector)[0].tolist()
    pred = classifier.predict(vector)[0]
    labels = classifier.classes_.tolist()
    return {"prediction": pred, "distribution": dist, "labels": labels}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)