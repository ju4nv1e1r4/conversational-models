import functools
import logging
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("compound_ai")

class Telemetry:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Telemetry, cls).__new__(cls)
            cls._instance._setup()
        return cls._instance

    def _setup(self):
        resource = Resource(attributes={
            "service.name": "compound-ai",
            "service.version": "0.1.0"
        })

        trace.set_tracer_provider(TracerProvider(resource=resource))
 
        otlp_exporter = OTLPSpanExporter(endpoint="http://jaeger:4317", insecure=True)

        trace.get_tracer_provider().add_span_processor(
            BatchSpanProcessor(otlp_exporter)
        )
        
        self.tracer = trace.get_tracer("compound.ai.tracer")

        RedisInstrumentor().instrument()

    def instrument_app(self, app):
        """Chamado no main.py para instrumentar o FastAPI"""
        FastAPIInstrumentor.instrument_app(app, tracer_provider=trace.get_tracer_provider())

telemetry = Telemetry()

def instrument(name=None):
    """
    Use @instrument() em cima de qualquer função para medir seu tempo e sucesso/erro automaticamente.
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            span_name = name or func.__name__
            with telemetry.tracer.start_as_current_span(span_name) as span:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR))
                    raise e

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            span_name = name or func.__name__
            with telemetry.tracer.start_as_current_span(span_name) as span:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR))
                    raise e

        if logging.getLogger().level == logging.DEBUG:
            pass
            
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator