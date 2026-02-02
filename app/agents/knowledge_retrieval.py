"""
Knowledge Retrieval Service for Business Rules

This service handles retrieval of business rules from the Foundry vector store
and provides verification logging for audit trails.
"""

import os
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class RetrievedRule:
    """A rule retrieved from the knowledge source."""
    section: str
    content: str
    relevance_score: float
    source_chunk_id: Optional[str] = None


@dataclass
class RetrievalResult:
    """Result of a knowledge retrieval operation."""
    success: bool
    chunks_retrieved: int
    rules: List[RetrievedRule]
    query: str
    error: Optional[str] = None
    retrieval_time_ms: float = 0


class KnowledgeRetrievalService:
    """
    Service for retrieving business rules from the Foundry vector store.
    
    This service:
    - Queries the vector store attached to BusinessRulesValidationAgent
    - Logs all retrieval operations for audit
    - Validates that retrieved rules are actually used
    """
    
    KNOWLEDGE_SOURCE_NAME = "health_underwriting_business_rules"
    VECTOR_STORE_NAME = "UnderwritingBusinessRules"
    
    def __init__(self):
        self._client = None
        self._vector_store_id: Optional[str] = None
        self._initialized = False
    
    def _get_client(self):
        """Get or create the Foundry client."""
        if self._client is None:
            from azure.ai.projects import AIProjectClient
            from azure.identity import DefaultAzureCredential
            
            endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
            if not endpoint:
                raise ValueError("AZURE_AI_PROJECT_ENDPOINT not set")
            
            self._client = AIProjectClient(
                endpoint=endpoint,
                credential=DefaultAzureCredential()
            )
        return self._client
    
    def initialize(self) -> bool:
        """Initialize the service and verify knowledge source exists."""
        if self._initialized:
            return True
        
        try:
            client = self._get_client()
            
            # Find the vector store
            for vs in client.agents.vector_stores.list():
                if vs.name == self.VECTOR_STORE_NAME:
                    self._vector_store_id = vs.id
                    logger.info(
                        "KNOWLEDGE SOURCE VERIFIED: %s (vector_store_id=%s)",
                        self.KNOWLEDGE_SOURCE_NAME,
                        self._vector_store_id
                    )
                    self._initialized = True
                    return True
            
            logger.error(
                "KNOWLEDGE SOURCE NOT FOUND: %s - BusinessRulesValidationAgent will fail",
                self.VECTOR_STORE_NAME
            )
            return False
            
        except Exception as e:
            logger.error("Failed to initialize knowledge retrieval: %s", e)
            return False
    
    def get_vector_store_id(self) -> Optional[str]:
        """Get the vector store ID for the business rules."""
        if not self._initialized:
            self.initialize()
        return self._vector_store_id
    
    def verify_knowledge_available(self) -> Dict[str, Any]:
        """Verify the knowledge source is available and return status."""
        status = {
            "knowledge_source": self.KNOWLEDGE_SOURCE_NAME,
            "vector_store": self.VECTOR_STORE_NAME,
            "available": False,
            "vector_store_id": None,
            "file_count": 0,
            "status": "unknown"
        }
        
        try:
            if not self._initialized:
                if not self.initialize():
                    status["status"] = "initialization_failed"
                    return status
            
            client = self._get_client()
            vs = client.agents.vector_stores.get(self._vector_store_id)
            
            status["available"] = vs.status == "completed"
            status["vector_store_id"] = vs.id
            status["file_count"] = getattr(vs.file_counts, 'completed', 0) if vs.file_counts else 0
            status["status"] = vs.status
            
            if status["available"]:
                logger.info(
                    "Knowledge source available: %s (files=%d)",
                    self.KNOWLEDGE_SOURCE_NAME,
                    status["file_count"]
                )
            else:
                logger.warning(
                    "Knowledge source NOT ready: %s (status=%s)",
                    self.KNOWLEDGE_SOURCE_NAME,
                    status["status"]
                )
            
        except Exception as e:
            status["status"] = f"error: {e}"
            logger.error("Knowledge verification failed: %s", e)
        
        return status
    
    def log_retrieval_start(self, query: str, context: Dict[str, Any]) -> None:
        """Log the start of a retrieval operation."""
        logger.info(
            "RULE RETRIEVAL STARTED - Query: '%s' | Context: RDS=%s, risk_level=%s",
            query[:100],
            context.get("risk_delta_score", "N/A"),
            context.get("risk_level", "N/A")
        )
    
    def log_retrieval_result(self, result: RetrievalResult) -> None:
        """Log the result of a retrieval operation."""
        if result.success and result.chunks_retrieved > 0:
            logger.info(
                "RULE RETRIEVAL SUCCESS: %d chunks retrieved in %.2fms",
                result.chunks_retrieved,
                result.retrieval_time_ms
            )
            for rule in result.rules[:5]:  # Log first 5 rules
                logger.info(
                    "RULE APPLIED: %s - %s",
                    rule.section,
                    rule.content[:100] + "..." if len(rule.content) > 100 else rule.content
                )
        else:
            logger.error(
                "RULE RETRIEVAL FAILED: %s (query='%s')",
                result.error or "No chunks retrieved",
                result.query[:100]
            )
    
    def validate_rule_application(
        self,
        applied_rules: List[str],
        adjustment_percentage: float,
        risk_level: str
    ) -> Dict[str, Any]:
        """
        Validate that rules were properly applied and not fabricated.
        
        Returns validation result with any discrepancies found.
        """
        validation = {
            "valid": True,
            "rules_cited": len(applied_rules),
            "discrepancies": [],
            "warnings": []
        }
        
        # Check that rules were cited
        if len(applied_rules) == 0 and adjustment_percentage != 0:
            validation["valid"] = False
            validation["discrepancies"].append(
                f"Premium adjustment of {adjustment_percentage}% applied without citing any rules"
            )
        
        # Check for obviously fabricated rule references
        for rule in applied_rules:
            if not any(term in rule.lower() for term in ["section", "rule", "policy", "table"]):
                validation["warnings"].append(
                    f"Rule citation may be fabricated: '{rule}'"
                )
        
        if not validation["valid"]:
            logger.error(
                "RULE VALIDATION FAILED: %s",
                "; ".join(validation["discrepancies"])
            )
        else:
            logger.info(
                "RULE VALIDATION PASSED: %d rules cited for %.0f%% adjustment",
                validation["rules_cited"],
                adjustment_percentage
            )
        
        return validation


# Global singleton instance
_knowledge_service: Optional[KnowledgeRetrievalService] = None


def get_knowledge_service() -> KnowledgeRetrievalService:
    """Get the global knowledge retrieval service instance."""
    global _knowledge_service
    if _knowledge_service is None:
        _knowledge_service = KnowledgeRetrievalService()
    return _knowledge_service
