from fastapi import APIRouter

from app.api.api_v1.endpoints import tags

api_router = APIRouter()
api_router.include_router(tags.router)
