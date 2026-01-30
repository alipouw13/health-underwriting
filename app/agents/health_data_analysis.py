"""
HealthDataAnalysisAgent - Analyze simulated Apple Health data for risk signals

Agent Definition (from /.github/underwriting_agents.yaml):
---------------------------------------------------------
agent_id: HealthDataAnalysisAgent
purpose: Analyze simulated Apple Health data to extract health risk signals
inputs:
  health_metrics: object
  patient_profile: object
outputs:
  risk_indicators: list
  summary: string
tools_used:
  - medical-mcp-simulator
evaluation_criteria:
  - signal_accuracy
  - explainability
failure_modes:
  - missing_data
  - inconsistent_metrics
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import Field

from data.mock.schemas import (
    HealthMetrics,
    PatientProfile,
    RiskIndicator,
    RiskLevel,
)
from app.agents.base import (
    BaseUnderwritingAgent,
    AgentInput,
    AgentOutput,
)


# =============================================================================
# INPUT/OUTPUT SCHEMAS
# =============================================================================

class HealthDataAnalysisInput(AgentInput):
    """Input schema for HealthDataAnalysisAgent."""
    
    health_metrics: HealthMetrics = Field(..., description="Health metrics from medical-mcp-simulator")
    patient_profile: PatientProfile = Field(..., description="Patient profile with demographics and history")


class HealthDataAnalysisOutput(AgentOutput):
    """Output schema for HealthDataAnalysisAgent."""
    
    risk_indicators: List[RiskIndicator] = Field(..., description="List of identified risk indicators")
    summary: str = Field(..., description="Human-readable summary of health risk analysis")


# =============================================================================
# AGENT IMPLEMENTATION
# =============================================================================

class HealthDataAnalysisAgent(BaseUnderwritingAgent[HealthDataAnalysisInput, HealthDataAnalysisOutput]):
    """
    Analyze simulated Apple Health data to extract health risk signals.
    
    This agent processes health metrics (activity, heart rate, sleep, trends)
    and patient profile data to identify risk indicators for underwriting.
    
    Tools Used:
        - medical-mcp-simulator: Provides simulated health data
    
    Evaluation Criteria:
        - signal_accuracy: Risk signals correctly identify health patterns
        - explainability: Risk indicators include clear explanations
    
    Failure Modes:
        - missing_data: Insufficient health data for analysis
        - inconsistent_metrics: Conflicting or impossible metric values
    """
    
    agent_id = "HealthDataAnalysisAgent"
    purpose = "Analyze simulated Apple Health data to extract health risk signals"
    tools_used = ["medical-mcp-simulator"]
    evaluation_criteria = ["signal_accuracy", "explainability"]
    failure_modes = ["missing_data", "inconsistent_metrics"]
    
    # Risk thresholds (configurable)
    RESTING_HR_LOW_MAX = 65
    RESTING_HR_MODERATE_MAX = 75
    RESTING_HR_HIGH_MAX = 85
    
    DAILY_STEPS_LOW_MIN = 8000
    DAILY_STEPS_MODERATE_MIN = 5000
    DAILY_STEPS_HIGH_MIN = 3000
    
    SLEEP_HOURS_LOW_MIN = 7.0
    SLEEP_HOURS_MODERATE_MIN = 6.0
    SLEEP_HOURS_HIGH_MIN = 5.0
    
    @property
    def input_type(self) -> type[HealthDataAnalysisInput]:
        return HealthDataAnalysisInput
    
    @property
    def output_type(self) -> type[HealthDataAnalysisOutput]:
        return HealthDataAnalysisOutput
    
    async def _execute(self, validated_input: HealthDataAnalysisInput) -> HealthDataAnalysisOutput:
        """
        Analyze health metrics and produce risk indicators.
        
        Analysis covers:
        1. Activity metrics (steps, exercise frequency)
        2. Heart rate metrics (resting HR, HRV, anomalies)
        3. Sleep metrics (duration, quality)
        4. Trend analysis
        5. Medical history correlation
        """
        health_metrics = validated_input.health_metrics
        patient_profile = validated_input.patient_profile
        
        risk_indicators: List[RiskIndicator] = []
        indicator_count = 0
        
        # Analyze activity metrics
        if health_metrics.activity:
            activity_indicators = self._analyze_activity(health_metrics.activity, indicator_count)
            risk_indicators.extend(activity_indicators)
            indicator_count += len(activity_indicators)
        
        # Analyze heart rate metrics
        if health_metrics.heart_rate:
            hr_indicators = self._analyze_heart_rate(
                health_metrics.heart_rate, 
                patient_profile,
                indicator_count
            )
            risk_indicators.extend(hr_indicators)
            indicator_count += len(hr_indicators)
        
        # Analyze sleep metrics
        if health_metrics.sleep:
            sleep_indicators = self._analyze_sleep(health_metrics.sleep, indicator_count)
            risk_indicators.extend(sleep_indicators)
            indicator_count += len(sleep_indicators)
        
        # Analyze trends
        if health_metrics.trends:
            trend_indicators = self._analyze_trends(health_metrics.trends, indicator_count)
            risk_indicators.extend(trend_indicators)
            indicator_count += len(trend_indicators)
        
        # Generate summary
        summary = self._generate_summary(risk_indicators, health_metrics, patient_profile)
        
        return HealthDataAnalysisOutput(
            agent_id=self.agent_id,
            risk_indicators=risk_indicators,
            summary=summary,
        )
    
    def _analyze_activity(
        self, 
        activity, 
        start_id: int
    ) -> List[RiskIndicator]:
        """Analyze activity metrics for risk signals."""
        indicators = []
        
        if activity.daily_steps_avg is not None:
            steps = activity.daily_steps_avg
            
            if steps >= self.DAILY_STEPS_LOW_MIN:
                risk_level = RiskLevel.LOW
                explanation = f"Good activity level with {steps:,} average daily steps."
            elif steps >= self.DAILY_STEPS_MODERATE_MIN:
                risk_level = RiskLevel.MODERATE
                explanation = f"Moderate activity level with {steps:,} average daily steps. Consider increasing to 8,000+."
            elif steps >= self.DAILY_STEPS_HIGH_MIN:
                risk_level = RiskLevel.HIGH
                explanation = f"Low activity level with {steps:,} average daily steps. Sedentary lifestyle increases health risks."
            else:
                risk_level = RiskLevel.VERY_HIGH
                explanation = f"Very low activity with only {steps:,} average daily steps. Significant sedentary risk factor."
            
            indicators.append(RiskIndicator(
                indicator_id=f"IND-ACT-{start_id + len(indicators) + 1:03d}",
                category="activity",
                indicator_name="Daily Step Count",
                risk_level=risk_level,
                confidence=self._calculate_data_confidence(activity.days_with_data, activity.measurement_period_days),
                metric_value=float(steps),
                metric_unit="steps/day",
                threshold_exceeded=f"steps < {self.DAILY_STEPS_MODERATE_MIN}" if risk_level != RiskLevel.LOW else None,
                explanation=explanation,
            ))
        
        # Exercise frequency analysis
        if activity.weekly_exercise_sessions is not None:
            sessions = activity.weekly_exercise_sessions
            
            if sessions >= 3:
                risk_level = RiskLevel.LOW
                explanation = f"Regular exercise habit with {sessions} sessions per week."
            elif sessions >= 1:
                risk_level = RiskLevel.MODERATE
                explanation = f"Infrequent exercise with only {sessions} session(s) per week."
            else:
                risk_level = RiskLevel.HIGH
                explanation = "No recorded exercise sessions. Lack of structured exercise is a risk factor."
            
            indicators.append(RiskIndicator(
                indicator_id=f"IND-ACT-{start_id + len(indicators) + 1:03d}",
                category="activity",
                indicator_name="Weekly Exercise Frequency",
                risk_level=risk_level,
                confidence=self._calculate_data_confidence(activity.days_with_data, activity.measurement_period_days),
                metric_value=float(sessions),
                metric_unit="sessions/week",
                threshold_exceeded="sessions < 3" if risk_level != RiskLevel.LOW else None,
                explanation=explanation,
            ))
        
        return indicators
    
    def _analyze_heart_rate(
        self, 
        heart_rate, 
        patient_profile: PatientProfile,
        start_id: int
    ) -> List[RiskIndicator]:
        """Analyze heart rate metrics for risk signals."""
        indicators = []
        
        # Resting heart rate analysis
        if heart_rate.resting_hr_avg is not None:
            rhr = heart_rate.resting_hr_avg
            age = patient_profile.demographics.age
            
            # Age-adjusted thresholds (older adults may have slightly higher normal RHR)
            age_adjustment = max(0, (age - 40) * 0.5)  # Add 0.5 bpm per year over 40
            
            if rhr <= self.RESTING_HR_LOW_MAX + age_adjustment:
                risk_level = RiskLevel.LOW
                explanation = f"Healthy resting heart rate of {rhr} bpm for age {age}."
            elif rhr <= self.RESTING_HR_MODERATE_MAX + age_adjustment:
                risk_level = RiskLevel.MODERATE
                explanation = f"Slightly elevated resting heart rate of {rhr} bpm. May indicate fitness opportunity."
            elif rhr <= self.RESTING_HR_HIGH_MAX + age_adjustment:
                risk_level = RiskLevel.HIGH
                explanation = f"Elevated resting heart rate of {rhr} bpm. Associated with increased cardiovascular risk."
            else:
                risk_level = RiskLevel.VERY_HIGH
                explanation = f"High resting heart rate of {rhr} bpm. Significant cardiovascular risk indicator."
            
            # Adjust if patient has known hypertension
            if patient_profile.medical_history.has_hypertension and risk_level == RiskLevel.MODERATE:
                risk_level = RiskLevel.HIGH
                explanation += " Risk elevated due to hypertension history."
            
            indicators.append(RiskIndicator(
                indicator_id=f"IND-HR-{start_id + len(indicators) + 1:03d}",
                category="heart_rate",
                indicator_name="Resting Heart Rate",
                risk_level=risk_level,
                confidence=self._calculate_data_confidence(heart_rate.days_with_data, heart_rate.measurement_period_days),
                metric_value=float(rhr),
                metric_unit="bpm",
                threshold_exceeded=f"rhr > {self.RESTING_HR_MODERATE_MAX}" if risk_level not in [RiskLevel.LOW, RiskLevel.MODERATE] else None,
                explanation=explanation,
            ))
        
        # Heart rate variability (HRV) analysis
        if heart_rate.hrv_avg_ms is not None:
            hrv = heart_rate.hrv_avg_ms
            
            # Lower HRV is associated with higher health risks
            if hrv >= 50:
                risk_level = RiskLevel.LOW
                explanation = f"Healthy HRV of {hrv:.1f}ms indicates good autonomic nervous system function."
            elif hrv >= 30:
                risk_level = RiskLevel.MODERATE
                explanation = f"Moderate HRV of {hrv:.1f}ms. Consider stress management."
            else:
                risk_level = RiskLevel.HIGH
                explanation = f"Low HRV of {hrv:.1f}ms. May indicate stress or cardiovascular concern."
            
            indicators.append(RiskIndicator(
                indicator_id=f"IND-HR-{start_id + len(indicators) + 1:03d}",
                category="heart_rate",
                indicator_name="Heart Rate Variability",
                risk_level=risk_level,
                confidence=self._calculate_data_confidence(heart_rate.days_with_data, heart_rate.measurement_period_days),
                metric_value=hrv,
                metric_unit="ms",
                threshold_exceeded="hrv < 30" if risk_level == RiskLevel.HIGH else None,
                explanation=explanation,
            ))
        
        # Irregular rhythm events
        if heart_rate.irregular_rhythm_events is not None and heart_rate.irregular_rhythm_events > 0:
            events = heart_rate.irregular_rhythm_events
            
            if events <= 2:
                risk_level = RiskLevel.MODERATE
                explanation = f"Detected {events} irregular rhythm event(s). May warrant monitoring."
            elif events <= 5:
                risk_level = RiskLevel.HIGH
                explanation = f"Detected {events} irregular rhythm events. Medical review recommended."
            else:
                risk_level = RiskLevel.VERY_HIGH
                explanation = f"Detected {events} irregular rhythm events. Significant arrhythmia concern."
            
            indicators.append(RiskIndicator(
                indicator_id=f"IND-HR-{start_id + len(indicators) + 1:03d}",
                category="heart_rate",
                indicator_name="Irregular Rhythm Events",
                risk_level=risk_level,
                confidence=0.9,  # Device-detected events have high confidence
                metric_value=float(events),
                metric_unit="events",
                threshold_exceeded="events > 0",
                explanation=explanation,
            ))
        
        return indicators
    
    def _analyze_sleep(
        self, 
        sleep, 
        start_id: int
    ) -> List[RiskIndicator]:
        """Analyze sleep metrics for risk signals."""
        indicators = []
        
        # Sleep duration analysis
        if sleep.avg_sleep_duration_hours is not None:
            duration = sleep.avg_sleep_duration_hours
            
            if duration >= self.SLEEP_HOURS_LOW_MIN:
                risk_level = RiskLevel.LOW
                explanation = f"Healthy sleep duration averaging {duration:.1f} hours per night."
            elif duration >= self.SLEEP_HOURS_MODERATE_MIN:
                risk_level = RiskLevel.MODERATE
                explanation = f"Borderline sleep duration of {duration:.1f} hours. 7+ hours recommended."
            elif duration >= self.SLEEP_HOURS_HIGH_MIN:
                risk_level = RiskLevel.HIGH
                explanation = f"Insufficient sleep averaging {duration:.1f} hours. Sleep deprivation increases health risks."
            else:
                risk_level = RiskLevel.VERY_HIGH
                explanation = f"Severely insufficient sleep at {duration:.1f} hours. Significant health concern."
            
            indicators.append(RiskIndicator(
                indicator_id=f"IND-SLP-{start_id + len(indicators) + 1:03d}",
                category="sleep",
                indicator_name="Average Sleep Duration",
                risk_level=risk_level,
                confidence=self._calculate_data_confidence(sleep.nights_with_data, sleep.measurement_period_days),
                metric_value=duration,
                metric_unit="hours",
                threshold_exceeded=f"hours < {self.SLEEP_HOURS_LOW_MIN}" if risk_level != RiskLevel.LOW else None,
                explanation=explanation,
            ))
        
        # Sleep efficiency
        if sleep.sleep_efficiency_pct is not None:
            efficiency = sleep.sleep_efficiency_pct
            
            if efficiency >= 85:
                risk_level = RiskLevel.LOW
                explanation = f"Good sleep efficiency at {efficiency:.0f}%."
            elif efficiency >= 75:
                risk_level = RiskLevel.MODERATE
                explanation = f"Fair sleep efficiency at {efficiency:.0f}%. May indicate sleep quality issues."
            else:
                risk_level = RiskLevel.HIGH
                explanation = f"Poor sleep efficiency at {efficiency:.0f}%. Sleep disorder screening may be warranted."
            
            indicators.append(RiskIndicator(
                indicator_id=f"IND-SLP-{start_id + len(indicators) + 1:03d}",
                category="sleep",
                indicator_name="Sleep Efficiency",
                risk_level=risk_level,
                confidence=self._calculate_data_confidence(sleep.nights_with_data, sleep.measurement_period_days),
                metric_value=efficiency,
                metric_unit="%",
                threshold_exceeded="efficiency < 85%" if risk_level != RiskLevel.LOW else None,
                explanation=explanation,
            ))
        
        return indicators
    
    def _analyze_trends(
        self, 
        trends, 
        start_id: int
    ) -> List[RiskIndicator]:
        """Analyze health trends for risk signals."""
        indicators = []
        
        # Overall health trajectory
        if trends.overall_health_trajectory:
            trajectory = trends.overall_health_trajectory.lower()
            
            if trajectory == "positive":
                risk_level = RiskLevel.LOW
                explanation = "Positive health trajectory indicates improving health behaviors."
            elif trajectory == "neutral":
                risk_level = RiskLevel.MODERATE
                explanation = "Stable health trajectory. No significant improvements or declines."
            else:  # negative
                risk_level = RiskLevel.HIGH
                explanation = "Declining health trajectory. Worsening health metrics over time."
            
            indicators.append(RiskIndicator(
                indicator_id=f"IND-TRD-{start_id + len(indicators) + 1:03d}",
                category="combined",
                indicator_name="Overall Health Trajectory",
                risk_level=risk_level,
                confidence=0.75,  # Trend analysis has inherent uncertainty
                metric_value=None,
                metric_unit=None,
                threshold_exceeded="trajectory = negative" if risk_level == RiskLevel.HIGH else None,
                explanation=explanation,
            ))
        
        # Significant changes
        if trends.significant_changes:
            for change in trends.significant_changes:
                # Treat significant changes as moderate concern requiring review
                indicators.append(RiskIndicator(
                    indicator_id=f"IND-TRD-{start_id + len(indicators) + 1:03d}",
                    category="combined",
                    indicator_name="Significant Health Change",
                    risk_level=RiskLevel.MODERATE,
                    confidence=0.7,
                    metric_value=None,
                    metric_unit=None,
                    threshold_exceeded="significant change detected",
                    explanation=f"Significant change detected: {change}",
                ))
        
        return indicators
    
    def _calculate_data_confidence(self, days_with_data: int, measurement_period: int) -> float:
        """Calculate confidence score based on data completeness."""
        if measurement_period == 0:
            return 0.0
        
        coverage = days_with_data / measurement_period
        
        # Scale confidence: 90%+ coverage = 0.95 confidence, linearly down to 0.5 at 30% coverage
        if coverage >= 0.9:
            return 0.95
        elif coverage >= 0.7:
            return 0.85
        elif coverage >= 0.5:
            return 0.75
        elif coverage >= 0.3:
            return 0.6
        else:
            return 0.5
    
    def _generate_summary(
        self, 
        indicators: List[RiskIndicator],
        health_metrics: HealthMetrics,
        patient_profile: PatientProfile
    ) -> str:
        """Generate human-readable summary of health risk analysis."""
        if not indicators:
            return "Insufficient data to generate health risk analysis. No metrics available for evaluation."
        
        # Count risk levels
        risk_counts = {level: 0 for level in RiskLevel}
        for ind in indicators:
            risk_counts[ind.risk_level] += 1
        
        # Determine overall assessment
        if risk_counts[RiskLevel.VERY_HIGH] > 0 or risk_counts[RiskLevel.DECLINE] > 0:
            overall = "HIGH RISK"
        elif risk_counts[RiskLevel.HIGH] > 0:
            overall = "ELEVATED RISK"
        elif risk_counts[RiskLevel.MODERATE] > 0:
            overall = "MODERATE RISK"
        else:
            overall = "LOW RISK"
        
        # Build summary
        summary_parts = [
            f"Health Risk Analysis for patient {health_metrics.patient_id}",
            f"Overall Assessment: {overall}",
            f"Identified {len(indicators)} risk indicators:",
            f"  - Low: {risk_counts[RiskLevel.LOW]}",
            f"  - Moderate: {risk_counts[RiskLevel.MODERATE]}",
            f"  - High: {risk_counts[RiskLevel.HIGH]}",
            f"  - Very High: {risk_counts[RiskLevel.VERY_HIGH]}",
        ]
        
        # Add key concerns
        high_risk_items = [ind for ind in indicators if ind.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]]
        if high_risk_items:
            summary_parts.append("\nKey Concerns:")
            for item in high_risk_items[:3]:  # Top 3 concerns
                summary_parts.append(f"  - {item.indicator_name}: {item.explanation}")
        
        return "\n".join(summary_parts)
