from app.api.api_v1.endpoints import tags
from fastapi import APIRouter

api_router = APIRouter()
api_router.include_router(tags.router)
