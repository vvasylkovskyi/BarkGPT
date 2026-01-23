from fastapi.responses import JSONResponse
from typing import Any, Optional
from fastapi import status

def handle_response(
    data: Optional[Any] = None,
    error: Optional[str] = None,
    status_code: int = status.HTTP_200_OK,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "success": error is None,
            "data": data if error is None else None,
            "error": error,
        },
    )
