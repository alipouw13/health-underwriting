"""
MCP Server Stubs for Underwriting Agents

SIMULATED INTERFACES - NOT FOR PRODUCTION USE

This module provides stubbed MCP (Model Context Protocol) server interfaces
that simulate the tool interactions defined in /.github/underwriting_agents.yaml.

MCP Server Mapping:
------------------
| Server Name               | Purpose                                        | Used By                          |
|---------------------------|------------------------------------------------|----------------------------------|
| medical-mcp-simulator     | Simulates Apple Health data                    | HealthDataAnalysisAgent          |
| policy-rule-engine        | Provides policy rules and thresholds           | PolicyRiskAgent                  |
| underwriting-rules-mcp    | Business rules and regulatory validation       | BusinessRulesValidationAgent     |
| data-quality-analyzer     | Data quality and completeness analysis         | DataQualityConfidenceAgent       |
| fairness-checker          | Bias detection and fairness analysis           | BiasAndFairnessAgent             |
| language-generator        | Natural language generation                    | CommunicationAgent               |
| trace-logger              | Audit logging and tracing                      | AuditAndTraceAgent               |

These stubs follow patterns from:
- https://github.com/sunanhe/awesome-medical-mcp-servers

Health Metrics Supported (per YAML constraints):
- activity
- heart_rate
- sleep
- trends

NO OTHER HEALTH METRICS SHOULD BE INVENTED.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from data.mock.schemas import (
    ActivityMetrics,
    HealthMetrics,
    HealthTrends,
    HeartRateMetrics,
    PatientProfile,
    PolicyRule,
    PolicyRuleSet,
    RiskLevel,
    RiskThreshold,
    SleepMetrics,
)
from data.mock.fixtures import (
    get_sample_patient_profiles,
    get_healthy_patient_metrics,
    get_moderate_risk_metrics,
    get_high_risk_metrics,
    get_standard_policy_rules,
)


# =============================================================================
# BASE MCP SERVER STUB
# =============================================================================

class BaseMCPServerStub(ABC):
    """
    Base class for all MCP server stubs.
    
    Provides common interface for tool invocations that would
    normally connect to external MCP servers.
    """
    
    server_name: str
    version: str = "1.0.0-stub"
    
    def __init__(self):
        self.logger = logging.getLogger(f"mcp.stub.{self.server_name}")
        self._call_count = 0
    
    @abstractmethod
    async def call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an MCP method call.
        
        Args:
            method: The MCP method to invoke
            params: Method parameters
            
        Returns:
            Method response
        """
        ...
    
    def _log_call(self, method: str, params: Dict[str, Any]) -> None:
        """Log an MCP call for debugging."""
        self._call_count += 1
        self.logger.debug(
            f"[STUB] {self.server_name}.{method}() call #{self._call_count}"
        )


# =============================================================================
# MEDICAL MCP SIMULATOR
# =============================================================================

class MedicalMCPSimulator(BaseMCPServerStub):
    """
    Stub for medical-mcp-simulator.
    
    Simulates Apple Health data retrieval. In production, this would
    connect to an MCP server that interfaces with health data APIs.
    
    Supported Methods:
    - get_health_metrics: Retrieve patient health metrics
    - get_patient_profile: Retrieve patient profile
    """
    
    server_name = "medical-mcp-simulator"
    
    # Pre-loaded sample data
    _patient_profiles: Dict[str, PatientProfile] = {}
    _health_metrics: Dict[str, HealthMetrics] = {}
    
    def __init__(self):
        super().__init__()
        self._load_sample_data()
    
    def _load_sample_data(self):
        """Load sample data from fixtures."""
        profiles = get_sample_patient_profiles()
        for profile in profiles:
            self._patient_profiles[profile.patient_id] = profile
        
        # Create corresponding health metrics
        self._health_metrics["PAT-001"] = get_healthy_patient_metrics()
        self._health_metrics["PAT-002"] = get_moderate_risk_metrics()
        self._health_metrics["PAT-003"] = get_high_risk_metrics()
    
    async def call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP method calls."""
        self._log_call(method, params)
        
        if method == "get_health_metrics":
            return await self._get_health_metrics(params)
        elif method == "get_patient_profile":
            return await self._get_patient_profile(params)
        elif method == "list_available_metrics":
            return await self._list_available_metrics()
        else:
            return {"error": f"Unknown method: {method}", "stub": True}
    
    async def _get_health_metrics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate retrieving health metrics for a patient."""
        patient_id = params.get("patient_id")
        
        if not patient_id:
            return {"error": "patient_id required", "stub": True}
        
        metrics = self._health_metrics.get(patient_id)
        
        if metrics:
            return {
                "success": True,
                "data": metrics.model_dump(),
                "stub": True,
            }
        else:
            # Generate default minimal metrics for unknown patients
            return {
                "success": True,
                "data": {
                    "patient_id": patient_id,
                    "data_source": "apple_health_simulated",
                    "activity": None,
                    "heart_rate": None,
                    "sleep": None,
                    "trends": None,
                },
                "stub": True,
            }
    
    async def _get_patient_profile(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate retrieving patient profile."""
        patient_id = params.get("patient_id")
        
        if not patient_id:
            return {"error": "patient_id required", "stub": True}
        
        profile = self._patient_profiles.get(patient_id)
        
        if profile:
            return {
                "success": True,
                "data": profile.model_dump(),
                "stub": True,
            }
        else:
            return {
                "success": False,
                "error": f"Patient {patient_id} not found",
                "stub": True,
            }
    
    async def _list_available_metrics(self) -> Dict[str, Any]:
        """List the supported health metric types."""
        return {
            "metrics": ["activity", "heart_rate", "sleep", "trends"],
            "stub": True,
            "note": "These are the ONLY supported metrics per YAML constraints",
        }


# =============================================================================
# POLICY RULE ENGINE
# =============================================================================

class PolicyRuleEngineStub(BaseMCPServerStub):
    """
    Stub for policy-rule-engine.
    
    Provides policy rules and risk thresholds for underwriting decisions.
    
    Supported Methods:
    - get_policy_rules: Retrieve rules for a policy type
    - evaluate_metric: Evaluate a single metric against rules
    """
    
    server_name = "policy-rule-engine"
    
    def __init__(self):
        super().__init__()
        self._policy_rules = get_standard_policy_rules()
    
    async def call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP method calls."""
        self._log_call(method, params)
        
        if method == "get_policy_rules":
            return await self._get_policy_rules(params)
        elif method == "evaluate_metric":
            return await self._evaluate_metric(params)
        else:
            return {"error": f"Unknown method: {method}", "stub": True}
    
    async def _get_policy_rules(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get policy rules for a given policy type."""
        policy_type = params.get("policy_type", "term_life")
        
        return {
            "success": True,
            "data": self._policy_rules.model_dump(),
            "stub": True,
        }
    
    async def _evaluate_metric(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single metric against policy rules."""
        metric_name = params.get("metric_name")
        metric_value = params.get("metric_value")
        
        if not metric_name or metric_value is None:
            return {"error": "metric_name and metric_value required", "stub": True}
        
        # Find applicable rules
        applicable_rules = [
            r for r in self._policy_rules.rules 
            if any(t.metric_name == metric_name for t in r.thresholds)
        ]
        
        return {
            "success": True,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "applicable_rules": [r.rule_id for r in applicable_rules],
            "stub": True,
        }


# =============================================================================
# UNDERWRITING RULES MCP
# =============================================================================

class UnderwritingRulesMCPStub(BaseMCPServerStub):
    """
    Stub for underwriting-rules-mcp.
    
    Provides business rules and regulatory compliance validation.
    
    Supported Methods:
    - validate_premium_adjustment: Validate a premium adjustment
    - get_regulatory_requirements: Get applicable regulations
    """
    
    server_name = "underwriting-rules-mcp"
    
    # Regulatory limits
    MAX_PREMIUM_INCREASE_PCT = 150.0
    MAX_DISCOUNT_PCT = 40.0
    
    async def call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP method calls."""
        self._log_call(method, params)
        
        if method == "validate_premium_adjustment":
            return await self._validate_premium_adjustment(params)
        elif method == "get_regulatory_requirements":
            return await self._get_regulatory_requirements(params)
        else:
            return {"error": f"Unknown method: {method}", "stub": True}
    
    async def _validate_premium_adjustment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a premium adjustment against business rules."""
        adjustment_pct = params.get("adjustment_percentage", 0)
        base_premium = params.get("base_premium", 0)
        
        violations = []
        
        if adjustment_pct > self.MAX_PREMIUM_INCREASE_PCT:
            violations.append(
                f"Adjustment {adjustment_pct}% exceeds maximum {self.MAX_PREMIUM_INCREASE_PCT}%"
            )
        
        if adjustment_pct < -self.MAX_DISCOUNT_PCT:
            violations.append(
                f"Discount {abs(adjustment_pct)}% exceeds maximum {self.MAX_DISCOUNT_PCT}%"
            )
        
        return {
            "success": len(violations) == 0,
            "violations": violations,
            "max_increase": self.MAX_PREMIUM_INCREASE_PCT,
            "max_discount": self.MAX_DISCOUNT_PCT,
            "stub": True,
        }
    
    async def _get_regulatory_requirements(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get regulatory requirements for a state/region."""
        region = params.get("region", "default")
        
        return {
            "region": region,
            "requirements": [
                "Premium adjustments must be actuarially justified",
                "Age-based adjustments must follow state guidelines",
                "Bias check required before approval",
                "Data quality assessment mandatory",
            ],
            "stub": True,
        }


# =============================================================================
# DATA QUALITY ANALYZER
# =============================================================================

class DataQualityAnalyzerStub(BaseMCPServerStub):
    """
    Stub for data-quality-analyzer.
    
    Provides data quality and completeness analysis.
    
    Supported Methods:
    - analyze_coverage: Analyze data coverage
    - check_freshness: Check data freshness
    - detect_anomalies: Detect data anomalies
    """
    
    server_name = "data-quality-analyzer"
    
    async def call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP method calls."""
        self._log_call(method, params)
        
        if method == "analyze_coverage":
            return await self._analyze_coverage(params)
        elif method == "check_freshness":
            return await self._check_freshness(params)
        elif method == "detect_anomalies":
            return await self._detect_anomalies(params)
        else:
            return {"error": f"Unknown method: {method}", "stub": True}
    
    async def _analyze_coverage(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data coverage for health metrics."""
        days_with_data = params.get("days_with_data", 0)
        measurement_period = params.get("measurement_period", 90)
        
        if measurement_period == 0:
            coverage = 0.0
        else:
            coverage = (days_with_data / measurement_period) * 100
        
        return {
            "coverage_pct": round(coverage, 1),
            "days_with_data": days_with_data,
            "measurement_period": measurement_period,
            "quality_level": (
                "excellent" if coverage >= 90 else
                "good" if coverage >= 70 else
                "fair" if coverage >= 50 else
                "poor" if coverage >= 30 else
                "insufficient"
            ),
            "stub": True,
        }
    
    async def _check_freshness(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check how recent the data is."""
        last_recorded = params.get("last_recorded_date")
        
        if last_recorded:
            if isinstance(last_recorded, str):
                last_date = date.fromisoformat(last_recorded)
            else:
                last_date = last_recorded
            
            days_old = (date.today() - last_date).days
        else:
            days_old = None
        
        return {
            "days_since_last_record": days_old,
            "is_stale": days_old is not None and days_old > 14,
            "freshness_level": (
                "fresh" if days_old is not None and days_old <= 3 else
                "recent" if days_old is not None and days_old <= 7 else
                "aging" if days_old is not None and days_old <= 14 else
                "stale" if days_old is not None else
                "unknown"
            ),
            "stub": True,
        }
    
    async def _detect_anomalies(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in health data."""
        # Stub implementation - would use statistical analysis in production
        return {
            "anomalies_detected": 0,
            "anomaly_details": [],
            "stub": True,
            "note": "Anomaly detection is stubbed - no real analysis performed",
        }


# =============================================================================
# FAIRNESS CHECKER
# =============================================================================

class FairnessCheckerStub(BaseMCPServerStub):
    """
    Stub for fairness-checker.
    
    Provides bias detection and fairness analysis.
    
    Supported Methods:
    - check_age_bias: Check for age-based discrimination
    - check_protected_attributes: Check usage of protected attributes
    - calculate_fairness_score: Calculate overall fairness score
    """
    
    server_name = "fairness-checker"
    
    PROTECTED_ATTRIBUTES = ["age", "sex", "race", "religion", "national_origin", "disability"]
    
    async def call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP method calls."""
        self._log_call(method, params)
        
        if method == "check_age_bias":
            return await self._check_age_bias(params)
        elif method == "check_protected_attributes":
            return await self._check_protected_attributes(params)
        elif method == "calculate_fairness_score":
            return await self._calculate_fairness_score(params)
        else:
            return {"error": f"Unknown method: {method}", "stub": True}
    
    async def _check_age_bias(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check for age-based discrimination."""
        age = params.get("age", 0)
        adjustment_pct = params.get("adjustment_pct", 0)
        
        # Simple check: flag high adjustments for seniors
        concern = age >= 65 and adjustment_pct > 30
        
        return {
            "age": age,
            "adjustment_pct": adjustment_pct,
            "concern_flagged": concern,
            "reason": "High adjustment for senior applicant" if concern else None,
            "stub": True,
        }
    
    async def _check_protected_attributes(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check if protected attributes are being used."""
        used_attributes = params.get("attributes_used", [])
        
        violations = [
            attr for attr in used_attributes 
            if attr.lower() in self.PROTECTED_ATTRIBUTES
        ]
        
        return {
            "attributes_checked": used_attributes,
            "protected_attributes_used": violations,
            "compliant": len(violations) == 0,
            "stub": True,
        }
    
    async def _calculate_fairness_score(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall fairness score."""
        # Stub implementation - would use ML fairness metrics in production
        return {
            "fairness_score": 0.95,
            "methodology": "stubbed",
            "stub": True,
            "note": "Real implementation would use statistical parity, equalized odds, etc.",
        }


# =============================================================================
# LANGUAGE GENERATOR
# =============================================================================

class LanguageGeneratorStub(BaseMCPServerStub):
    """
    Stub for language-generator.
    
    Provides natural language generation for communications.
    
    Supported Methods:
    - generate_message: Generate a message
    - assess_readability: Assess text readability
    - check_tone: Check message tone
    """
    
    server_name = "language-generator"
    
    async def call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP method calls."""
        self._log_call(method, params)
        
        if method == "generate_message":
            return await self._generate_message(params)
        elif method == "assess_readability":
            return await self._assess_readability(params)
        elif method == "check_tone":
            return await self._check_tone(params)
        else:
            return {"error": f"Unknown method: {method}", "stub": True}
    
    async def _generate_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a message (stubbed - returns template)."""
        message_type = params.get("type", "generic")
        context = params.get("context", {})
        
        return {
            "message": f"[Stubbed {message_type} message for context: {context}]",
            "stub": True,
            "note": "Real implementation would use LLM for generation",
        }
    
    async def _assess_readability(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Assess text readability."""
        text = params.get("text", "")
        
        # Simple word/sentence count
        words = text.split()
        sentences = text.count(".") + text.count("!") + text.count("?")
        
        return {
            "word_count": len(words),
            "sentence_count": max(1, sentences),
            "avg_words_per_sentence": len(words) / max(1, sentences),
            "stub": True,
        }
    
    async def _check_tone(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check message tone."""
        text = params.get("text", "").lower()
        
        positive_words = ["pleased", "congratulations", "welcome", "thank"]
        negative_words = ["unfortunately", "unable", "declined", "rejected"]
        
        positive_count = sum(1 for w in positive_words if w in text)
        negative_count = sum(1 for w in negative_words if w in text)
        
        if positive_count > negative_count:
            tone = "positive"
        elif negative_count > positive_count:
            tone = "negative"
        else:
            tone = "neutral"
        
        return {
            "tone": tone,
            "positive_indicators": positive_count,
            "negative_indicators": negative_count,
            "stub": True,
        }


# =============================================================================
# TRACE LOGGER
# =============================================================================

class TraceLoggerStub(BaseMCPServerStub):
    """
    Stub for trace-logger.
    
    Provides audit logging and tracing capabilities.
    
    Supported Methods:
    - log_event: Log an audit event
    - get_trace: Get trace for a workflow
    - verify_integrity: Verify audit log integrity
    """
    
    server_name = "trace-logger"
    
    _logs: List[Dict[str, Any]] = []
    
    async def call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP method calls."""
        self._log_call(method, params)
        
        if method == "log_event":
            return await self._log_event(params)
        elif method == "get_trace":
            return await self._get_trace(params)
        elif method == "verify_integrity":
            return await self._verify_integrity(params)
        else:
            return {"error": f"Unknown method: {method}", "stub": True}
    
    async def _log_event(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Log an audit event."""
        event = {
            "event_id": str(uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **params,
        }
        self._logs.append(event)
        
        return {
            "success": True,
            "event_id": event["event_id"],
            "stub": True,
        }
    
    async def _get_trace(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get trace for a workflow."""
        workflow_id = params.get("workflow_id")
        
        matching_events = [
            e for e in self._logs 
            if e.get("workflow_id") == workflow_id
        ]
        
        return {
            "workflow_id": workflow_id,
            "event_count": len(matching_events),
            "events": matching_events,
            "stub": True,
        }
    
    async def _verify_integrity(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Verify audit log integrity."""
        return {
            "integrity_verified": True,
            "total_events": len(self._logs),
            "stub": True,
            "note": "Stubbed verification - no real integrity check performed",
        }


# =============================================================================
# MCP REGISTRY
# =============================================================================

class MCPServerRegistry:
    """
    Registry for MCP server stubs.
    
    Provides a central place to get MCP server instances.
    """
    
    _servers: Dict[str, BaseMCPServerStub] = {}
    
    @classmethod
    def get_server(cls, server_name: str) -> BaseMCPServerStub:
        """Get or create an MCP server stub."""
        if server_name not in cls._servers:
            cls._servers[server_name] = cls._create_server(server_name)
        return cls._servers[server_name]
    
    @classmethod
    def _create_server(cls, server_name: str) -> BaseMCPServerStub:
        """Create the appropriate MCP server stub."""
        server_map = {
            "medical-mcp-simulator": MedicalMCPSimulator,
            "policy-rule-engine": PolicyRuleEngineStub,
            "underwriting-rules-mcp": UnderwritingRulesMCPStub,
            "data-quality-analyzer": DataQualityAnalyzerStub,
            "fairness-checker": FairnessCheckerStub,
            "language-generator": LanguageGeneratorStub,
            "trace-logger": TraceLoggerStub,
        }
        
        server_class = server_map.get(server_name)
        if server_class:
            return server_class()
        
        # Return a generic stub for unknown servers
        class GenericStub(BaseMCPServerStub):
            server_name = server_name
            
            async def call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
                self._log_call(method, params)
                return {"stub": True, "server": self.server_name, "method": method}
        
        return GenericStub()
    
    @classmethod
    def list_servers(cls) -> List[str]:
        """List all available MCP server stubs."""
        return [
            "medical-mcp-simulator",
            "policy-rule-engine",
            "underwriting-rules-mcp",
            "data-quality-analyzer",
            "fairness-checker",
            "language-generator",
            "trace-logger",
        ]
