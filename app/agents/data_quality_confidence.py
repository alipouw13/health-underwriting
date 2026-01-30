"""
DataQualityConfidenceAgent - Assess reliability and completeness of health data

Agent Definition (from /.github/underwriting_agents.yaml):
---------------------------------------------------------
agent_id: DataQualityConfidenceAgent
purpose: Assess reliability and completeness of health data
inputs:
  health_metrics: object
outputs:
  confidence_score: number
  quality_flags: list
tools_used:
  - data-quality-analyzer
evaluation_criteria:
  - coverage
  - freshness
failure_modes:
  - insufficient_data
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import List, Optional
from pydantic import Field

from data.mock.schemas import (
    HealthMetrics,
    QualityFlag,
    DataQualityLevel,
)
from app.agents.base import (
    BaseUnderwritingAgent,
    AgentInput,
    AgentOutput,
)


# =============================================================================
# INPUT/OUTPUT SCHEMAS
# =============================================================================

class DataQualityConfidenceInput(AgentInput):
    """Input schema for DataQualityConfidenceAgent."""
    
    health_metrics: HealthMetrics = Field(..., description="Health metrics to assess for quality")


class DataQualityConfidenceOutput(AgentOutput):
    """Output schema for DataQualityConfidenceAgent."""
    
    confidence_score: float = Field(..., ge=0, le=1, description="Overall confidence score (0-1)")
    quality_flags: List[QualityFlag] = Field(default_factory=list, description="Quality issues identified")
    data_quality_level: DataQualityLevel = Field(..., description="Overall data quality classification")
    coverage_metrics: dict = Field(default_factory=dict, description="Coverage statistics by metric type")
    freshness_assessment: str = Field(..., description="Assessment of data freshness")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for improving data quality")


# =============================================================================
# AGENT IMPLEMENTATION
# =============================================================================

class DataQualityConfidenceAgent(BaseUnderwritingAgent[DataQualityConfidenceInput, DataQualityConfidenceOutput]):
    """
    Assess reliability and completeness of health data.
    
    This agent evaluates the quality of health metrics data to determine
    how much confidence can be placed in risk assessments.
    
    Tools Used:
        - data-quality-analyzer: Provides quality analysis capabilities
    
    Evaluation Criteria:
        - coverage: What percentage of the measurement period has data
        - freshness: How recent is the latest data
    
    Failure Modes:
        - insufficient_data: Not enough data to make reliable assessment
    """
    
    agent_id = "DataQualityConfidenceAgent"
    purpose = "Assess reliability and completeness of health data"
    tools_used = ["data-quality-analyzer"]
    evaluation_criteria = ["coverage", "freshness"]
    failure_modes = ["insufficient_data"]
    
    # Quality thresholds
    MIN_COVERAGE_PCT = 30  # Minimum acceptable coverage
    GOOD_COVERAGE_PCT = 70  # Coverage for "good" quality
    EXCELLENT_COVERAGE_PCT = 90  # Coverage for "excellent" quality
    STALE_DATA_DAYS = 14  # Data older than this is considered stale
    MIN_DAYS_FOR_TREND = 30  # Minimum days needed for trend analysis
    
    @property
    def input_type(self) -> type[DataQualityConfidenceInput]:
        return DataQualityConfidenceInput
    
    @property
    def output_type(self) -> type[DataQualityConfidenceOutput]:
        return DataQualityConfidenceOutput
    
    async def _execute(self, validated_input: DataQualityConfidenceInput) -> DataQualityConfidenceOutput:
        """
        Assess data quality and calculate confidence score.
        
        Analysis covers:
        1. Coverage - percentage of days with data
        2. Freshness - recency of data
        3. Completeness - all metric types present
        4. Consistency - no impossible/conflicting values
        """
        health_metrics = validated_input.health_metrics
        
        quality_flags: List[QualityFlag] = []
        coverage_metrics: dict = {}
        recommendations: List[str] = []
        flag_count = 0
        
        # Initialize confidence at maximum, reduce based on issues
        base_confidence = 1.0
        confidence_penalties: List[float] = []
        
        # Assess activity data quality
        if health_metrics.activity:
            activity_coverage = self._calculate_coverage(
                health_metrics.activity.days_with_data,
                health_metrics.activity.measurement_period_days
            )
            coverage_metrics["activity"] = activity_coverage
            
            flags, penalty = self._assess_metric_quality(
                "activity",
                activity_coverage,
                health_metrics.activity.last_recorded_date,
                health_metrics.activity.measurement_period_days,
                flag_count
            )
            quality_flags.extend(flags)
            confidence_penalties.append(penalty)
            flag_count += len(flags)
        else:
            quality_flags.append(QualityFlag(
                flag_id=f"QF-{flag_count + 1:03d}",
                flag_type="missing_data",
                severity="warning",
                affected_metric="activity",
                description="No activity data available",
                confidence_impact=-0.2,
            ))
            confidence_penalties.append(0.2)
            recommendations.append("Request activity data from health device")
            flag_count += 1
        
        # Assess heart rate data quality
        if health_metrics.heart_rate:
            hr_coverage = self._calculate_coverage(
                health_metrics.heart_rate.days_with_data,
                health_metrics.heart_rate.measurement_period_days
            )
            coverage_metrics["heart_rate"] = hr_coverage
            
            flags, penalty = self._assess_metric_quality(
                "heart_rate",
                hr_coverage,
                health_metrics.heart_rate.last_recorded_date,
                health_metrics.heart_rate.measurement_period_days,
                flag_count
            )
            quality_flags.extend(flags)
            confidence_penalties.append(penalty)
            flag_count += len(flags)
            
            # Additional heart rate consistency checks
            if health_metrics.heart_rate.resting_hr_min and health_metrics.heart_rate.resting_hr_max:
                if health_metrics.heart_rate.resting_hr_min > health_metrics.heart_rate.resting_hr_max:
                    quality_flags.append(QualityFlag(
                        flag_id=f"QF-{flag_count + 1:03d}",
                        flag_type="inconsistent",
                        severity="critical",
                        affected_metric="heart_rate",
                        description="Min resting HR exceeds max resting HR - impossible values",
                        confidence_impact=-0.3,
                    ))
                    confidence_penalties.append(0.3)
                    flag_count += 1
        else:
            quality_flags.append(QualityFlag(
                flag_id=f"QF-{flag_count + 1:03d}",
                flag_type="missing_data",
                severity="warning",
                affected_metric="heart_rate",
                description="No heart rate data available",
                confidence_impact=-0.2,
            ))
            confidence_penalties.append(0.2)
            recommendations.append("Request heart rate data from health device")
            flag_count += 1
        
        # Assess sleep data quality
        if health_metrics.sleep:
            sleep_coverage = self._calculate_coverage(
                health_metrics.sleep.nights_with_data,
                health_metrics.sleep.measurement_period_days
            )
            coverage_metrics["sleep"] = sleep_coverage
            
            flags, penalty = self._assess_metric_quality(
                "sleep",
                sleep_coverage,
                health_metrics.sleep.last_recorded_date,
                health_metrics.sleep.measurement_period_days,
                flag_count
            )
            quality_flags.extend(flags)
            confidence_penalties.append(penalty)
            flag_count += len(flags)
            
            # Sleep percentage consistency check
            if health_metrics.sleep.deep_sleep_pct is not None and \
               health_metrics.sleep.rem_sleep_pct is not None and \
               health_metrics.sleep.light_sleep_pct is not None:
                total_pct = (
                    health_metrics.sleep.deep_sleep_pct +
                    health_metrics.sleep.rem_sleep_pct +
                    health_metrics.sleep.light_sleep_pct
                )
                if abs(total_pct - 100) > 5:  # Allow 5% tolerance
                    quality_flags.append(QualityFlag(
                        flag_id=f"QF-{flag_count + 1:03d}",
                        flag_type="inconsistent",
                        severity="warning",
                        affected_metric="sleep",
                        description=f"Sleep stage percentages sum to {total_pct}%, expected ~100%",
                        confidence_impact=-0.1,
                    ))
                    confidence_penalties.append(0.1)
                    flag_count += 1
        else:
            quality_flags.append(QualityFlag(
                flag_id=f"QF-{flag_count + 1:03d}",
                flag_type="missing_data",
                severity="warning",
                affected_metric="sleep",
                description="No sleep data available",
                confidence_impact=-0.15,
            ))
            confidence_penalties.append(0.15)
            recommendations.append("Request sleep data from health device")
            flag_count += 1
        
        # Assess trends data quality
        if health_metrics.trends:
            coverage_metrics["trends"] = 100.0  # Trends are derived, consider full coverage
            
            # Check if sufficient data for trend analysis
            min_period = min(
                health_metrics.activity.measurement_period_days if health_metrics.activity else 0,
                health_metrics.heart_rate.measurement_period_days if health_metrics.heart_rate else 0,
                health_metrics.sleep.measurement_period_days if health_metrics.sleep else 0,
            )
            
            if min_period < self.MIN_DAYS_FOR_TREND:
                quality_flags.append(QualityFlag(
                    flag_id=f"QF-{flag_count + 1:03d}",
                    flag_type="incomplete",
                    severity="info",
                    affected_metric="trends",
                    description=f"Only {min_period} days of data - trends may be unreliable (need {self.MIN_DAYS_FOR_TREND}+)",
                    confidence_impact=-0.1,
                ))
                confidence_penalties.append(0.1)
                flag_count += 1
        else:
            coverage_metrics["trends"] = 0.0
        
        # Calculate final confidence score
        total_penalty = sum(confidence_penalties)
        confidence_score = max(0.0, base_confidence - total_penalty)
        
        # Determine overall data quality level
        data_quality_level = self._determine_quality_level(confidence_score, quality_flags)
        
        # Assess freshness
        freshness_assessment = self._assess_freshness(health_metrics)
        
        # Generate recommendations based on issues
        if not recommendations:
            if confidence_score >= 0.9:
                recommendations.append("Data quality is excellent - no action needed")
            elif confidence_score >= 0.7:
                recommendations.append("Consider extending measurement period for more reliable trends")
            else:
                recommendations.append("Request additional health data to improve assessment confidence")
        
        return DataQualityConfidenceOutput(
            agent_id=self.agent_id,
            confidence_score=round(confidence_score, 3),
            quality_flags=quality_flags,
            data_quality_level=data_quality_level,
            coverage_metrics=coverage_metrics,
            freshness_assessment=freshness_assessment,
            recommendations=recommendations,
        )
    
    def _calculate_coverage(self, days_with_data: int, measurement_period: int) -> float:
        """Calculate coverage percentage."""
        if measurement_period == 0:
            return 0.0
        return round((days_with_data / measurement_period) * 100, 1)
    
    def _assess_metric_quality(
        self,
        metric_name: str,
        coverage: float,
        last_recorded: Optional[date],
        measurement_period: int,
        start_flag_id: int
    ) -> tuple[List[QualityFlag], float]:
        """
        Assess quality of a specific metric.
        
        Returns:
            Tuple of (quality_flags, total_penalty)
        """
        flags: List[QualityFlag] = []
        penalty = 0.0
        flag_id = start_flag_id
        
        # Coverage check
        if coverage < self.MIN_COVERAGE_PCT:
            flags.append(QualityFlag(
                flag_id=f"QF-{flag_id + 1:03d}",
                flag_type="incomplete",
                severity="critical",
                affected_metric=metric_name,
                description=f"Only {coverage:.0f}% coverage (minimum {self.MIN_COVERAGE_PCT}% required)",
                confidence_impact=-0.25,
            ))
            penalty += 0.25
            flag_id += 1
        elif coverage < self.GOOD_COVERAGE_PCT:
            flags.append(QualityFlag(
                flag_id=f"QF-{flag_id + 1:03d}",
                flag_type="incomplete",
                severity="warning",
                affected_metric=metric_name,
                description=f"Coverage at {coverage:.0f}% - recommend {self.GOOD_COVERAGE_PCT}%+ for reliable analysis",
                confidence_impact=-0.1,
            ))
            penalty += 0.1
            flag_id += 1
        
        # Freshness check
        if last_recorded:
            days_since_last = (date.today() - last_recorded).days
            if days_since_last > self.STALE_DATA_DAYS:
                flags.append(QualityFlag(
                    flag_id=f"QF-{flag_id + 1:03d}",
                    flag_type="stale_data",
                    severity="warning",
                    affected_metric=metric_name,
                    description=f"Last recording was {days_since_last} days ago (stale threshold: {self.STALE_DATA_DAYS} days)",
                    confidence_impact=-0.15,
                ))
                penalty += 0.15
                flag_id += 1
        
        # Measurement period check
        if measurement_period < self.MIN_DAYS_FOR_TREND:
            flags.append(QualityFlag(
                flag_id=f"QF-{flag_id + 1:03d}",
                flag_type="incomplete",
                severity="info",
                affected_metric=metric_name,
                description=f"Measurement period of {measurement_period} days is short for trend analysis",
                confidence_impact=-0.05,
            ))
            penalty += 0.05
        
        return flags, penalty
    
    def _determine_quality_level(
        self, 
        confidence: float, 
        flags: List[QualityFlag]
    ) -> DataQualityLevel:
        """Determine overall data quality level."""
        critical_flags = [f for f in flags if f.severity == "critical"]
        
        if critical_flags:
            return DataQualityLevel.POOR if confidence >= 0.3 else DataQualityLevel.INSUFFICIENT
        
        if confidence >= 0.9:
            return DataQualityLevel.EXCELLENT
        elif confidence >= 0.75:
            return DataQualityLevel.GOOD
        elif confidence >= 0.5:
            return DataQualityLevel.FAIR
        elif confidence >= 0.3:
            return DataQualityLevel.POOR
        else:
            return DataQualityLevel.INSUFFICIENT
    
    def _assess_freshness(self, health_metrics: HealthMetrics) -> str:
        """Generate freshness assessment string."""
        latest_dates = []
        
        if health_metrics.activity and health_metrics.activity.last_recorded_date:
            latest_dates.append(("activity", health_metrics.activity.last_recorded_date))
        if health_metrics.heart_rate and health_metrics.heart_rate.last_recorded_date:
            latest_dates.append(("heart_rate", health_metrics.heart_rate.last_recorded_date))
        if health_metrics.sleep and health_metrics.sleep.last_recorded_date:
            latest_dates.append(("sleep", health_metrics.sleep.last_recorded_date))
        
        if not latest_dates:
            return "No recording dates available - freshness unknown"
        
        most_recent_metric, most_recent_date = max(latest_dates, key=lambda x: x[1])
        oldest_metric, oldest_date = min(latest_dates, key=lambda x: x[1])
        
        days_since_recent = (date.today() - most_recent_date).days
        
        if days_since_recent == 0:
            freshness = "Very fresh - data recorded today"
        elif days_since_recent <= 3:
            freshness = f"Fresh - most recent data is {days_since_recent} day(s) old"
        elif days_since_recent <= 7:
            freshness = f"Recent - most recent data is {days_since_recent} days old"
        elif days_since_recent <= self.STALE_DATA_DAYS:
            freshness = f"Aging - most recent data is {days_since_recent} days old"
        else:
            freshness = f"Stale - most recent data is {days_since_recent} days old"
        
        return freshness
