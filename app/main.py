from fastapi.concurrency import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from collections.abc import AsyncGenerator

from app.routes.routes import create_router
from app.context.app_context import AppContext
from app.settings.app import get_settings


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    AppContext.initialize(get_settings())
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(create_router())
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
