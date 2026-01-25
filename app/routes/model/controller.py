from fastapi import APIRouter, Request

from app.context.app_context import AppContext


model_router = APIRouter(prefix="/model", tags=["Model"])


@model_router.post("/generate")
async def generate(request: Request):
    body = await request.json()
    prompt = body.get("prompt")
    generator = AppContext.get_instance().get_bark_generator()
    text: str = generator.generate(prompt)

    return {"message": text}
