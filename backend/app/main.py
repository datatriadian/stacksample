from fastapi import FastAPI

from app.api.api import api_router
from app.core.config import API_V1_STR

app = FastAPI()

app.include_router(api_router, prefix=API_V1_STR)
