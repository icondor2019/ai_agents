from fastapi import APIRouter

from app.controllers import (
    health_controller,
    hf_agent_controller
)

api_router = APIRouter()
api_router.include_router(health_controller.api)
api_router.include_router(hf_agent_controller.api)
