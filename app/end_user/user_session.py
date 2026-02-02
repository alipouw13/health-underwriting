"""
End User Session Management

DEMO ONLY - NOT FOR PRODUCTION USE
This module provides simple in-memory session management for end-user demo flow.
No real authentication is performed.
"""

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional
import threading


@dataclass
class EndUserProfile:
    """End user profile created during mock login."""
    first_name: str
    last_name: str
    date_of_birth: date
    biological_sex: str = "unknown"  # 'male', 'female', 'unknown'
    
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"
    
    @property
    def age(self) -> int:
        return (date.today() - self.date_of_birth).days // 365


@dataclass
class EndUserSession:
    """
    End user session state.
    
    Tracks the user's progress through the demo flow:
    1. Login (creates user profile)
    2. Apple Health connection (synthetic data)
    3. Application creation
    4. Risk analysis
    """
    session_id: str
    user_id: str
    profile: EndUserProfile
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Flow state
    apple_health_connected: bool = False
    apple_health_data: Optional[Dict[str, Any]] = None
    apple_health_consent_timestamp: Optional[datetime] = None
    
    # Application state
    application_id: Optional[str] = None
    application_created_at: Optional[datetime] = None
    
    # Analysis state
    risk_analysis_completed: bool = False
    risk_analysis_workflow_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for API responses."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "profile": {
                "first_name": self.profile.first_name,
                "last_name": self.profile.last_name,
                "full_name": self.profile.full_name,
                "date_of_birth": self.profile.date_of_birth.isoformat(),
                "age": self.profile.age,
                "biological_sex": self.profile.biological_sex,
            },
            "created_at": self.created_at.isoformat(),
            "apple_health_connected": self.apple_health_connected,
            "apple_health_consent_timestamp": (
                self.apple_health_consent_timestamp.isoformat() 
                if self.apple_health_consent_timestamp else None
            ),
            "application_id": self.application_id,
            "application_created_at": (
                self.application_created_at.isoformat()
                if self.application_created_at else None
            ),
            "risk_analysis_completed": self.risk_analysis_completed,
            "risk_analysis_workflow_id": self.risk_analysis_workflow_id,
            "flow_state": self._get_flow_state(),
        }
    
    def _get_flow_state(self) -> str:
        """Determine current state in the user flow."""
        if self.risk_analysis_completed:
            return "completed"
        if self.application_id:
            return "application_created"
        if self.apple_health_connected:
            return "health_connected"
        return "logged_in"


class EndUserSessionStore:
    """
    In-memory session store for end-user sessions.
    
    DEMO ONLY - Sessions are lost on server restart.
    For production, use a proper session store (Redis, database, etc.)
    """
    
    def __init__(self):
        self._sessions: Dict[str, EndUserSession] = {}
        self._user_sessions: Dict[str, str] = {}  # user_id -> session_id
        self._lock = threading.Lock()
    
    def create_session(
        self,
        first_name: str,
        last_name: str,
        date_of_birth: date,
        biological_sex: str = "unknown",
    ) -> EndUserSession:
        """
        Create a new end-user session.
        
        Args:
            first_name: User's first name
            last_name: User's last name
            date_of_birth: User's date of birth
            biological_sex: Biological sex for actuarial purposes
            
        Returns:
            New EndUserSession object
        """
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        user_id = f"eu_{uuid.uuid4().hex[:8]}"
        
        profile = EndUserProfile(
            first_name=first_name,
            last_name=last_name,
            date_of_birth=date_of_birth,
            biological_sex=biological_sex,
        )
        
        session = EndUserSession(
            session_id=session_id,
            user_id=user_id,
            profile=profile,
        )
        
        with self._lock:
            self._sessions[session_id] = session
            self._user_sessions[user_id] = session_id
        
        return session
    
    def get_session(self, session_id: str) -> Optional[EndUserSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)
    
    def get_session_by_user_id(self, user_id: str) -> Optional[EndUserSession]:
        """Get a session by user ID."""
        session_id = self._user_sessions.get(user_id)
        if session_id:
            return self._sessions.get(session_id)
        return None
    
    def update_session(self, session: EndUserSession) -> None:
        """Update a session in the store."""
        with self._lock:
            if session.session_id in self._sessions:
                self._sessions[session.session_id] = session
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                self._user_sessions.pop(session.user_id, None)
                return True
            return False
    
    def list_sessions(self) -> list:
        """List all active sessions (for debugging)."""
        return list(self._sessions.values())


# Global session store instance
user_session_store = EndUserSessionStore()
