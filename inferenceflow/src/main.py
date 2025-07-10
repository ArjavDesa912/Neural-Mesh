from fastapi import FastAPI
from src.api import routes
from prometheus_client import start_http_server
from src.utils.config import settings

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    start_http_server(settings.PROMETHEUS_PORT)

app.include_router(routes.router)
