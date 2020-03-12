import uvicorn
from fastapi import Body, FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

import pickle
import bz2

with open('models/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
with bz2.open('models/classifier.pkl.bz2', 'rb') as f:
    classifier = pickle.load(f)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def read_root():
    return RedirectResponse("static/index.html")


class Article(BaseModel):
    title: str = Field("", title='Title of the article')
    text: str = Field("", title='Text of the article')
    mscs: str = Field("", title='Comma separated list of space separated MSCs in referenced articles.')


@app.post("/classify/")
async def read_item(*, article: Article = Body(...)):
    vector = encoder.transform([article.title + article.text + article.mscs])
    dist = classifier.predict_proba(vector)[0].tolist()
    pred = classifier.predict(vector)[0]
    labels = classifier.classes_.tolist()
    return {"prediction": pred, "distribution": dist, "labels": labels}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
