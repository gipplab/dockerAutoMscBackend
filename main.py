import uvicorn
from fastapi import Body, FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

import pickle
import bz2

example = {
    "title": "On a generalization of the Rogers generating function.",
    "text": "We derive a generalization of the Rogers generating function for the continuous \\("
            "q\\)-ultraspherical/Rogers polynomials whose coefficient is a \\(_2\\phi_1\\). From that expansion, "
            "we derive corresponding specialization and limit transition expansions for the continuous \\(q\\)-Hermite,"
            " continuous \\(q\\)-Legendre, Laguerre, and Chebyshev polynomials of the first kind. Using a generalized "
            "expansion of the Rogers generating function in terms of the Askey-Wilson polynomials by Ismail \\& "
            "Simeonov whose coefficient is a \\(_8\\phi_7\\), we derive corresponding generalized expansions for the "
            "Wilson, continuous \\(q\\)-Jacobi, and Jacobi polynomials. By comparing the coefficients of the "
            "Askey-Wilson expansion to our continuous \\(q\\)-ultraspherical/Rogers expansion, we derive a new "
            "quadratic transformation for basic hypergeometric functions which relates an \\(_8\\phi_7\\) to a \\("
            "_2\\phi_1\\). We also obtain several definite integral representations which correspond to the above "
            "mentioned expansions through the use of orthogonality.",
    "mscs": "33C55 33C45 33D45, 33D45 39A10, 35A08 31B30 31C12 33C05 42A16 35C10"
}

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
    title: str = Field("", title='Title')
    text: str = Field("", title='Text')
    mscs: str = Field("", title='List of MSCs',
                      desciption='Comma separated list of space separated MSCs in referenced articles.')


@app.post("/classify/")
async def read_item(article: Article = Body(..., example=example)):
    vector = encoder.transform([article.title + article.text + article.mscs])
    dist = classifier.predict_proba(vector)[0].tolist()
    pred = classifier.predict(vector)[0]
    labels = classifier.classes_.tolist()
    return {"prediction": pred, "distribution": dist, "labels": labels}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
