"""
Azure AI Foundry Tracing Integration

This module configures OpenTelemetry tracing to send traces to Azure Application Insights,
which can then be viewed in the Azure AI Foundry portal under the Tracing section.

To enable tracing:
1. In Azure AI Foundry portal, navigate to your project > Tracing
2. Connect an Application Insights resource (or create one)
3. Copy the connection string
4. Set the APPLICATIONINSIGHTS_CONNECTION_STRING environment variable

Environment Variables:
- APPLICATIONINSIGHTS_CONNECTION_STRING: Connection string from Application Insights
- OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: Set to "true" to capture message content (optional)
- FOUNDRY_PROJECT_ENDPOINT: Your Foundry project endpoint (optional, for auto-discovery)

References:
- https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/trace-application
"""

import logging
import os
from typing import Optional

logger = logging.getLogger("underwriting_assistant.tracing")

# Global tracer instance
_tracer = None
_tracing_enabled = False


def configure_tracing(connection_string: Optional[str] = None) -> bool:
    """
    Configure Azure Monitor OpenTelemetry tracing.
    
    Args:
        connection_string: Application Insights connection string. 
                          If not provided, reads from APPLICATIONINSIGHTS_CONNECTION_STRING env var.
    
    Returns:
        True if tracing was configured successfully, False otherwise.
    """
    global _tracer, _tracing_enabled
    
    # Get connection string from parameter or environment
    conn_str = connection_string or os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    
    if not conn_str:
        logger.warning(
            "No Application Insights connection string found. "
            "Set APPLICATIONINSIGHTS_CONNECTION_STRING to enable Foundry portal tracing. "
            "Traces will only be available locally."
        )
        return False
    
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
        from opentelemetry import trace
        
        # Configure Azure Monitor exporter
        configure_azure_monitor(connection_string=conn_str)
        
        # Optionally instrument OpenAI SDK for automatic tracing of LLM calls
        try:
            from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
            OpenAIInstrumentor().instrument()
            logger.info("✅ OpenAI SDK instrumented for automatic tracing")
        except ImportError:
            logger.warning(
                "opentelemetry-instrumentation-openai-v2 not installed. "
                "OpenAI calls will not be automatically traced."
            )
        except Exception as e:
            logger.warning(f"Could not instrument OpenAI SDK: {e}")
        
        # Get the tracer instance
        _tracer = trace.get_tracer("underwriting_assistant")
        _tracing_enabled = True
        
        logger.info("✅ Azure Application Insights tracing configured successfully")
        logger.info("   View traces in Azure AI Foundry portal > Your Project > Tracing")
        
        return True
        
    except ImportError as e:
        logger.warning(
            f"Azure Monitor OpenTelemetry packages not installed: {e}. "
            "Install with: pip install azure-monitor-opentelemetry opentelemetry-instrumentation-openai-v2"
        )
        return False
    except Exception as e:
        logger.error(f"Failed to configure Azure Monitor tracing: {e}")
        return False


def configure_console_tracing() -> bool:
    """
    Configure tracing to output to console (useful for local development/debugging).
    
    Returns:
        True if console tracing was configured successfully.
    """
    global _tracer, _tracing_enabled
    
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
        
        # Instrument OpenAI SDK if available
        try:
            from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
            OpenAIInstrumentor().instrument()
        except ImportError:
            pass
        
        # Configure console exporter
        span_exporter = ConsoleSpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
        trace.set_tracer_provider(tracer_provider)
        
        _tracer = trace.get_tracer("underwriting_assistant")
        _tracing_enabled = True
        
        logger.info("✅ Console tracing configured (traces will print to stdout)")
        return True
        
    except ImportError as e:
        logger.warning(f"OpenTelemetry SDK not installed: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to configure console tracing: {e}")
        return False


def get_tracer():
    """
    Get the configured tracer instance.
    
    Returns:
        The OpenTelemetry tracer if configured, or a no-op tracer.
    """
    global _tracer
    
    if _tracer is not None:
        return _tracer
    
    # Return a no-op tracer if not configured
    try:
        from opentelemetry import trace
        return trace.get_tracer("underwriting_assistant")
    except ImportError:
        return None


def is_tracing_enabled() -> bool:
    """Check if tracing is enabled and configured."""
    return _tracing_enabled


def trace_span(name: str):
    """
    Decorator to trace a function as a span.
    
    Usage:
        @trace_span("my_operation")
        def my_function():
            ...
    """
    def decorator(func):
        tracer = get_tracer()
        if tracer is None:
            return func
        return tracer.start_as_current_span(name)(func)
    return decorator


def add_span_attribute(key: str, value) -> None:
    """
    Add an attribute to the current span.
    
    Args:
        key: Attribute name
        value: Attribute value
    """
    try:
        from opentelemetry import trace
        current_span = trace.get_current_span()
        if current_span:
            current_span.set_attribute(key, value)
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Could not add span attribute: {e}")


def add_span_event(name: str, attributes: dict = None) -> None:
    """
    Add an event to the current span.
    
    Args:
        name: Event name
        attributes: Optional event attributes
    """
    try:
        from opentelemetry import trace
        current_span = trace.get_current_span()
        if current_span:
            current_span.add_event(name, attributes=attributes or {})
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Could not add span event: {e}")
