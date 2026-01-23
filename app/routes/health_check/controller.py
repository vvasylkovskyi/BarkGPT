from fastapi import APIRouter, status
from app.http.response import handle_response

health_check_router = APIRouter(prefix="/health-check")


@health_check_router.get("/")
async def health_check():
    return handle_response(data={"status": "OK"}, status_code=status.HTTP_200_OK)
