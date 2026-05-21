from fastapi.concurrency import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from collections.abc import AsyncGenerator

from app.routes.routes import create_router
from app.context.app_context import AppContext
from app.settings.app import get_settings
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    AppContext.initialize(get_settings())
    yield


# The service name is the label that scopes all your alerts and dashboards.
# Must match OTEL_SERVICE_NAME if you set it via environment variable.
resource = Resource(attributes={"service.name": "my-api"})

exporter = OTLPMetricExporter(
    endpoint="http://host.docker.internal:4317",  # replace with your Alloy host
    insecure=True,
)

reader = PeriodicExportingMetricReader(exporter, export_interval_millis=15_000)
provider = MeterProvider(resource=resource, metric_readers=[reader])
metrics.set_meter_provider(provider)

app = FastAPI(lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app)

app.include_router(create_router())
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
