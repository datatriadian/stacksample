from app.api.api import api_router
from app.core.classifier import load_classifier, load_vectorizer
from app.core.config import API_V1_STR
from fastapi import FastAPI

app = FastAPI()


@app.on_event("startup")
def cache_models() -> None:
    load_classifier()
    load_vectorizer()


app.include_router(api_router, prefix=API_V1_STR)
