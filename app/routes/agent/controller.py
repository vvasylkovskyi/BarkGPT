import logging
from fastapi import APIRouter, status, Request
from app.http.response import handle_response

from app.agent.sre_agent import SREAgent, TaskSubmittedResponse
from dataclasses import asdict

logger = logging.getLogger(__name__)


agent_router = APIRouter(prefix="/agent", tags=["Agent"])


@agent_router.post("/run", status_code=status.HTTP_200_OK)
async def run_agent(request: Request):

    agent = SREAgent()
    response: TaskSubmittedResponse = await agent.run()
    print(">>> Agent run response: ", response)
    return handle_response(
        data={"status": "OK", "response": asdict(response)},
        status_code=status.HTTP_200_OK,
    )
