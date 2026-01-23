from fastapi import APIRouter
from app.routes.health_check.controller import health_check_router
from app.routes.agent.controller import agent_router
from app.routes.model.controller import model_router


def create_router():
    router = APIRouter(prefix="/v1")
    router.include_router(model_router, tags=["Model"])
    router.include_router(health_check_router, tags=["Health"])
    router.include_router(agent_router, tags=["Agent"])
    return router
