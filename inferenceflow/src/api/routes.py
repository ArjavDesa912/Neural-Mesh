from fastapi import APIRouter
from src.api import gateway

router = APIRouter()

router.include_router(gateway.router)
