from fastapi.concurrency import asynccontextmanager
from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from collections.abc import AsyncGenerator

# from a2a.server.apps.jsonrpc import A2AFastAPIApplication
# from a2a.server.request_handlers import DefaultRequestHandler
# from a2a.types import AgentCapabilities, AgentCard, AgentProvider, AgentSkill
# from app.a2a_integration.task_store import in_memory_task_store
from app.routes.routes import create_router
from app.kafka.logs_agent_results_consumer_worker import (
    LogsAgentResultsConsumerWorker,
)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    # global stoi
    # global itos
    # global hf_model

    # checkpoint = torch.load(os.environ["MODEL_PATH"], map_location="cpu")
    # stoi = checkpoint["stoi"]
    # itos = checkpoint["itos"]
    # vocab_size = checkpoint["vocab_size"]

    # bark = BarkGPT(vocab_size)
    # bark.load_state_dict(checkpoint["model_state"])

    # config = BarkConfig(vocab_size=vocab_size)
    # hf_model = BarkHF(config, bark)
    # Initialize and start LogsAgentResultsWorker
    worker = LogsAgentResultsConsumerWorker()
    await worker.start()
    yield
    await worker.stop()


# agent_card = AgentCard(
#     name="Logs Agent",
#     description="Agents that retrieves logs",
#     version="0.1.0",
#     url="http://localhost:8007",
#     protocol_version="0.3.0",
#     skills=[
#         AgentSkill(
#             id="fetch-logs",
#             name="Fetch Logs",
#             description="""Fetch Logs""",
#             tags=[
#                 "monitoring",
#             ],
#             examples=[
#                 "Explain the Datadog trigger for this alert",
#             ],
#         ),
#     ],
#     default_input_modes=["text/plain"],
#     default_output_modes=["text/plain"],
#     capabilities=AgentCapabilities(streaming=True),
#     provider=AgentProvider(
#         organization="PagerDuty",
#         url="http://localhost:8007",
#     ),
# )


# handler = DefaultRequestHandler(
#     agent_executor=SREAgentExecutor(),
#     task_store=in_memory_task_store,
# )

# a2a = A2AFastAPIApplication(
#     agent_card=agent_card,
#     http_handler=handler,
# )


app = FastAPI(lifespan=lifespan)
# a2a.add_routes_to_app(app)

app.include_router(create_router())
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
