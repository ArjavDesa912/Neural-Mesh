from fastapi import APIRouter
from src.api.gateway import api_gateway

# Use the router from the API gateway
router = api_gateway.app
