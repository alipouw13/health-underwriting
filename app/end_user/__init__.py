"""
End User Module

Provides functionality for the end-user (applicant) flow that:
- Simulates Apple Health data connection
- Creates application objects using synthetic health data
- Routes to the SAME agent pipeline as underwriters

IMPORTANT: This is a DEMO implementation. No real Apple APIs or OAuth.
"""

from .apple_health_mock import AppleHealthMockData, generate_apple_health_data
from .user_session import EndUserSession, user_session_store
from .application_generator import generate_application_document, generate_and_extract_application

__all__ = [
    "AppleHealthMockData",
    "generate_apple_health_data",
    "EndUserSession",
    "user_session_store",
    "generate_application_document",
    "generate_and_extract_application",
]
