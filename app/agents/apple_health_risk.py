"""
AppleHealthRiskAgent - Calculate HealthKit Risk Score (HKRS) for Apple Health workflow

This agent evaluates Apple HealthKit data against the Apple Health underwriting
policies to produce a HealthKit Risk Score (HKRS) from 0-100.

The HKRS formula is:
    HKRS = (Σ weighted sub-scores) × Age Adjustment Factor

Sub-scores:
    - Activity Score (25%): Steps/day, trends
    - VO2 Max Score (20%): Cardiorespiratory fitness
    - Heart Health Score (20%): Resting HR, HRV
    - Sleep Health Score (15%): Duration, consistency
    - Body Composition Score (10%): BMI/weight trend
    - Mobility Score (10%): Walking speed, steadiness

Policies are loaded from: prompts/apple-health-underwriting-policies.json
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple
from pydantic import Field
from enum import Enum

from data.mock.schemas import (
    HealthMetrics,
    PatientProfile,
    RiskLevel,
)
from app.agents.base import (
    BaseUnderwritingAgent,
    AgentInput,
    AgentOutput,
)


# =============================================================================
# ENUMS
# =============================================================================

class HKRSBand(str, Enum):
    """HKRS classification bands."""
    EXCELLENT = "excellent"           # 85-100
    VERY_GOOD = "very_good"           # 70-84
    STANDARD_PLUS = "standard_plus"   # 55-69
    STANDARD = "standard"             # 40-54
    SUBSTANDARD = "substandard"       # 0-39


class DataQuality(str, Enum):
    """Data quality classification based on completeness."""
    HIGH = "high"       # >=80% completeness
    MEDIUM = "medium"   # 50-79%
    LOW = "low"         # <50%


# =============================================================================
# INPUT/OUTPUT SCHEMAS
# =============================================================================

class AppleHealthRiskInput(AgentInput):
    """Input schema for AppleHealthRiskAgent."""
    
    health_metrics: HealthMetrics = Field(..., description="Apple HealthKit metrics")
    patient_profile: PatientProfile = Field(..., description="Patient profile with age and demographics")


class SubScoreDetail(AgentOutput):
    """Detail for a single sub-score calculation."""
    
    name: str = Field(..., description="Sub-score name")
    raw_score: float = Field(..., description="Raw score before weighting")
    weight: float = Field(..., description="Weight applied to this score")
    weighted_score: float = Field(..., description="Score after weighting")
    max_points: int = Field(..., description="Maximum possible points")
    components: Dict[str, Any] = Field(default_factory=dict, description="Component breakdowns")
    notes: List[str] = Field(default_factory=list, description="Calculation notes")


class AppleHealthRiskOutput(AgentOutput):
    """Output schema for AppleHealthRiskAgent.
    
    Produces the HealthKit Risk Score (HKRS) and risk classification.
    """
    
    # Core HKRS output
    hkrs: float = Field(..., ge=0, le=100, description="HealthKit Risk Score (0-100)")
    hkrs_band: HKRSBand = Field(..., description="HKRS classification band")
    hkrs_band_description: str = Field(..., description="Human-readable band description")
    
    # Age adjustment
    age_adjustment_factor: float = Field(..., description="AAF applied based on age")
    age_bracket: str = Field(..., description="Age bracket used for AAF")
    
    # Sub-scores breakdown
    sub_scores: List[SubScoreDetail] = Field(default_factory=list, description="Individual sub-score details")
    raw_score_before_aaf: float = Field(..., description="Sum of weighted scores before AAF")
    
    # Data quality
    data_quality: DataQuality = Field(..., description="Data quality classification")
    data_quality_score: float = Field(..., description="Data completeness percentage")
    data_gaps: List[str] = Field(default_factory=list, description="Missing or insufficient data")
    
    # Risk class recommendation
    risk_class_recommendation: str = Field(..., description="Recommended risk class")
    adjustment_action: str = Field(..., description="Action based on HKRS band")
    
    # Decision fields (matching PolicyRiskAgent interface)
    approved: bool = Field(default=True, description="Whether application is approved")
    decision: str = Field(default="approved", description="Final decision")
    rationale: str = Field(default="", description="Explanation for the decision")
    referral_required: bool = Field(default=False, description="Whether manual review is required")
    
    # Explainability (required by policy)
    top_positive_drivers: List[str] = Field(default_factory=list, description="Top 3 positive factors")
    improvement_suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")
    summary_scorecard: str = Field(default="", description="Applicant-facing summary")


# =============================================================================
# AGENT IMPLEMENTATION
# =============================================================================

class AppleHealthRiskAgent(BaseUnderwritingAgent[AppleHealthRiskInput, AppleHealthRiskOutput]):
    """
    Calculate HealthKit Risk Score (HKRS) for Apple Health workflow.
    
    This agent processes Apple HealthKit data using the specialized
    scoring model defined in apple-health-underwriting-policies.json.
    
    Key Principles:
        - Behavior-based signals matter (activity, sleep, fitness)
        - Longitudinal trends (90-365 days) > point-in-time snapshots
        - HealthKit data is additive only (cannot auto-decline)
        - All decisions must be explainable
    """
    
    agent_id = "AppleHealthRiskAgent"
    purpose = "Calculate HealthKit Risk Score for Apple Health underwriting"
    tools_used = ["apple-health-policy-engine"]
    evaluation_criteria = ["hkrs_accuracy", "explainability", "consistency"]
    failure_modes = ["insufficient_data", "data_quality_low"]
    
    # Age Adjustment Factors (from policy)
    AAF_TABLE = {
        "18-34": 1.00,
        "35-44": 0.98,
        "45-54": 0.95,
        "55-64": 0.92,
        "65+": 0.88,
    }
    
    # Sub-score weights (from policy)
    SCORE_WEIGHTS = {
        "activity": 0.25,
        "vo2_max": 0.20,
        "heart_health": 0.20,
        "sleep_health": 0.15,
        "body_composition": 0.10,
        "mobility": 0.10,
    }
    
    # HKRS band thresholds
    HKRS_BANDS = {
        HKRSBand.EXCELLENT: (85, 100),
        HKRSBand.VERY_GOOD: (70, 84),
        HKRSBand.STANDARD_PLUS: (55, 69),
        HKRSBand.STANDARD: (40, 54),
        HKRSBand.SUBSTANDARD: (0, 39),
    }
    
    @property
    def input_type(self) -> type[AppleHealthRiskInput]:
        return AppleHealthRiskInput
    
    @property
    def output_type(self) -> type[AppleHealthRiskOutput]:
        return AppleHealthRiskOutput
    
    async def _execute(self, validated_input: AppleHealthRiskInput) -> AppleHealthRiskOutput:
        """
        Calculate HKRS from Apple HealthKit data.
        
        Process:
        1. Assess data quality and completeness
        2. Calculate each sub-score
        3. Apply weights and sum
        4. Apply Age Adjustment Factor
        5. Determine HKRS band and recommendations
        6. Generate explainability outputs
        """
        health_metrics = validated_input.health_metrics
        patient_profile = validated_input.patient_profile
        
        # Step 1: Assess data quality
        data_quality, dq_score, data_gaps = self._assess_data_quality(health_metrics)
        
        # Step 2: Get age bracket and AAF
        age = patient_profile.demographics.age if patient_profile.demographics else 35
        age_bracket = self._get_age_bracket(age)
        aaf = self.AAF_TABLE.get(age_bracket, 0.95)
        
        # Step 3: Calculate sub-scores
        sub_scores = []
        
        # Activity Score (25%)
        activity_score = self._calculate_activity_score(health_metrics)
        sub_scores.append(activity_score)
        
        # VO2 Max Score (20%)
        vo2_score = self._calculate_vo2_max_score(health_metrics, age)
        sub_scores.append(vo2_score)
        
        # Heart Health Score (20%)
        heart_score = self._calculate_heart_health_score(health_metrics, age)
        sub_scores.append(heart_score)
        
        # Sleep Health Score (15%)
        sleep_score = self._calculate_sleep_health_score(health_metrics)
        sub_scores.append(sleep_score)
        
        # Body Composition Score (10%)
        body_score = self._calculate_body_composition_score(health_metrics, patient_profile)
        sub_scores.append(body_score)
        
        # Mobility Score (10%)
        mobility_score = self._calculate_mobility_score(health_metrics, age)
        sub_scores.append(mobility_score)
        
        # Step 4: Sum weighted scores
        raw_score = sum(s.weighted_score for s in sub_scores)
        
        # Step 5: Apply AAF
        hkrs = round(raw_score * aaf, 1)
        hkrs = min(100, max(0, hkrs))  # Clamp to 0-100
        
        # Step 6: Determine HKRS band
        hkrs_band = self._get_hkrs_band(hkrs)
        band_desc = self._get_band_description(hkrs_band)
        
        # Step 7: Determine recommendations and actions
        risk_class, action = self._get_risk_class_recommendation(hkrs, hkrs_band, data_quality)
        
        # Step 8: Determine approval status
        approved, decision, referral_required = self._determine_decision(hkrs_band, data_quality)
        
        # Step 9: Generate explainability outputs
        top_drivers = self._get_top_positive_drivers(sub_scores)
        suggestions = self._get_improvement_suggestions(sub_scores, health_metrics)
        summary = self._generate_summary_scorecard(hkrs, hkrs_band, top_drivers)
        rationale = self._generate_rationale(hkrs, hkrs_band, sub_scores, data_quality)
        
        return AppleHealthRiskOutput(
            agent_id=self.agent_id,
            hkrs=hkrs,
            hkrs_band=hkrs_band,
            hkrs_band_description=band_desc,
            age_adjustment_factor=aaf,
            age_bracket=age_bracket,
            sub_scores=sub_scores,
            raw_score_before_aaf=round(raw_score, 1),
            data_quality=data_quality,
            data_quality_score=dq_score,
            data_gaps=data_gaps,
            risk_class_recommendation=risk_class,
            adjustment_action=action,
            approved=approved,
            decision=decision,
            rationale=rationale,
            referral_required=referral_required,
            top_positive_drivers=top_drivers,
            improvement_suggestions=suggestions,
            summary_scorecard=summary,
        )
    
    # =========================================================================
    # DATA QUALITY ASSESSMENT
    # =========================================================================
    
    def _assess_data_quality(self, metrics: HealthMetrics) -> Tuple[DataQuality, float, List[str]]:
        """Assess data quality based on completeness requirements."""
        gaps = []
        completeness_scores = []
        
        # Check steps data (min 120 days)
        if metrics.activity and metrics.activity.days_with_data:
            steps_completeness = min(100, (metrics.activity.days_with_data / 120) * 100)
            completeness_scores.append(steps_completeness)
            if metrics.activity.days_with_data < 120:
                gaps.append(f"Steps: {metrics.activity.days_with_data}/120 days (need more data)")
        else:
            completeness_scores.append(0)
            gaps.append("Steps: No activity data available")
        
        # Check sleep data (min 90 days)
        if metrics.sleep and metrics.sleep.nights_with_data:
            sleep_completeness = min(100, (metrics.sleep.nights_with_data / 90) * 100)
            completeness_scores.append(sleep_completeness)
            if metrics.sleep.nights_with_data < 90:
                gaps.append(f"Sleep: {metrics.sleep.nights_with_data}/90 nights (need more data)")
        else:
            completeness_scores.append(0)
            gaps.append("Sleep: No sleep data available")
        
        # Check heart rate data (min 60 days)
        if metrics.heart_rate and metrics.heart_rate.days_with_data:
            hr_completeness = min(100, (metrics.heart_rate.days_with_data / 60) * 100)
            completeness_scores.append(hr_completeness)
            if metrics.heart_rate.days_with_data < 60:
                gaps.append(f"Heart Rate: {metrics.heart_rate.days_with_data}/60 days (need more data)")
        else:
            completeness_scores.append(0)
            gaps.append("Heart Rate: No heart rate data available")
        
        # Calculate overall DQ score
        dq_score = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0
        
        # Determine quality level
        if dq_score >= 80:
            quality = DataQuality.HIGH
        elif dq_score >= 50:
            quality = DataQuality.MEDIUM
        else:
            quality = DataQuality.LOW
        
        return quality, round(dq_score, 1), gaps
    
    # =========================================================================
    # AGE BRACKET CALCULATION
    # =========================================================================
    
    def _get_age_bracket(self, age: int) -> str:
        """Determine age bracket for AAF lookup."""
        if age < 18:
            return "18-34"  # Treat under 18 as 18-34
        elif age <= 34:
            return "18-34"
        elif age <= 44:
            return "35-44"
        elif age <= 54:
            return "45-54"
        elif age <= 64:
            return "55-64"
        else:
            return "65+"
    
    # =========================================================================
    # SUB-SCORE CALCULATIONS
    # =========================================================================
    
    def _calculate_activity_score(self, metrics: HealthMetrics) -> SubScoreDetail:
        """Calculate Activity Score (25% weight, max 25 points)."""
        max_points = 25
        weight = self.SCORE_WEIGHTS["activity"]
        components = {}
        notes = []
        raw_score = 0
        
        if metrics.activity:
            # Steps per day scoring
            steps = metrics.activity.daily_steps_avg or 0
            if steps > 8000:
                steps_score = 25
                notes.append(f"Excellent step count: {steps:,}/day")
            elif steps >= 6000:
                steps_score = 18
                notes.append(f"Good step count: {steps:,}/day")
            elif steps >= 4000:
                steps_score = 10
                notes.append(f"Moderate step count: {steps:,}/day")
            else:
                steps_score = 0
                notes.append(f"Low step count: {steps:,}/day (target: 8,000+)")
            
            components["steps_per_day"] = {"value": steps, "score": steps_score}
            raw_score = steps_score
            
            # Trend modifier
            if metrics.trends:
                trend = metrics.trends.activity_trend_weekly or "stable"
                if trend == "improving":
                    raw_score = min(max_points, raw_score + 3)
                    components["trend_modifier"] = 3
                    notes.append("Activity trend: Improving (+3)")
                elif trend == "declining":
                    raw_score = max(0, raw_score - 3)
                    components["trend_modifier"] = -3
                    notes.append("Activity trend: Declining (-3)")
                else:
                    components["trend_modifier"] = 0
        else:
            notes.append("No activity data available")
        
        return SubScoreDetail(
            agent_id=self.agent_id,
            name="activity_score",
            raw_score=raw_score,
            weight=weight,
            weighted_score=round(raw_score * weight, 2),
            max_points=max_points,
            components=components,
            notes=notes,
        )
    
    def _calculate_vo2_max_score(self, metrics: HealthMetrics, age: int) -> SubScoreDetail:
        """Calculate VO2 Max Score (20% weight, max 20 points)."""
        max_points = 20
        weight = self.SCORE_WEIGHTS["vo2_max"]
        components = {}
        notes = []
        raw_score = 0
        
        # VO2 Max may come from fitness metrics (we'll estimate from activity if not available)
        vo2_max = None
        
        # Check if we have VO2 max data (would be in extended metrics)
        if hasattr(metrics, 'fitness') and metrics.fitness:
            vo2_max = getattr(metrics.fitness, 'vo2_max', None)
        
        if vo2_max:
            # Calculate percentile based on age-adjusted norms
            percentile = self._estimate_vo2_percentile(vo2_max, age)
            
            if percentile >= 75:
                raw_score = 20
                notes.append(f"Excellent VO2 Max: {vo2_max:.1f} (>75th percentile)")
            elif percentile >= 50:
                raw_score = 15
                notes.append(f"Good VO2 Max: {vo2_max:.1f} (50-74th percentile)")
            elif percentile >= 25:
                raw_score = 8
                notes.append(f"Fair VO2 Max: {vo2_max:.1f} (25-49th percentile)")
            else:
                raw_score = 0
                notes.append(f"Below average VO2 Max: {vo2_max:.1f} (<25th percentile)")
            
            components["vo2_max"] = {"value": vo2_max, "percentile": percentile, "score": raw_score}
        else:
            # Estimate from activity level if no VO2 data
            if metrics.activity and metrics.activity.daily_steps_avg:
                steps = metrics.activity.daily_steps_avg
                if steps > 10000:
                    raw_score = 15  # Assume good fitness
                    notes.append("VO2 Max: Estimated from high activity level")
                elif steps > 6000:
                    raw_score = 10
                    notes.append("VO2 Max: Estimated from moderate activity level")
                else:
                    raw_score = 5
                    notes.append("VO2 Max: Estimated from low activity level")
                components["estimated_from_activity"] = True
            else:
                notes.append("No VO2 Max data available")
        
        return SubScoreDetail(
            agent_id=self.agent_id,
            name="vo2_max_score",
            raw_score=raw_score,
            weight=weight,
            weighted_score=round(raw_score * weight, 2),
            max_points=max_points,
            components=components,
            notes=notes,
        )
    
    def _calculate_heart_health_score(self, metrics: HealthMetrics, age: int) -> SubScoreDetail:
        """Calculate Heart Health Score (20% weight, max 20 points)."""
        max_points = 20
        weight = self.SCORE_WEIGHTS["heart_health"]
        components = {}
        notes = []
        raw_score = 0
        
        if metrics.heart_rate:
            # Resting HR scoring (max 10 points)
            resting_hr = metrics.heart_rate.resting_hr_avg
            if resting_hr:
                if 50 <= resting_hr <= 70:
                    hr_score = 10
                    notes.append(f"Optimal resting HR: {resting_hr} bpm")
                elif 71 <= resting_hr <= 75:
                    hr_score = 7
                    notes.append(f"Good resting HR: {resting_hr} bpm")
                elif 76 <= resting_hr <= 80:
                    hr_score = 4
                    notes.append(f"Elevated resting HR: {resting_hr} bpm")
                elif resting_hr < 50:
                    hr_score = 5  # Athletically low
                    notes.append(f"Athletic resting HR: {resting_hr} bpm")
                else:
                    hr_score = 0
                    notes.append(f"High resting HR: {resting_hr} bpm (target: 50-70)")
                
                components["resting_hr"] = {"value": resting_hr, "score": hr_score}
                raw_score += hr_score
            
            # HRV scoring (max 10 points)
            hrv = metrics.heart_rate.hrv_avg_ms
            if hrv:
                # Age-adjusted HRV thresholds (simplified)
                hrv_60th = self._get_hrv_60th_percentile(age)
                
                if hrv >= hrv_60th:
                    hrv_score = 10
                    notes.append(f"Excellent HRV: {hrv:.1f}ms (≥60th percentile)")
                elif hrv >= hrv_60th * 0.7:
                    hrv_score = 6
                    notes.append(f"Good HRV: {hrv:.1f}ms")
                else:
                    hrv_score = 0
                    notes.append(f"Below average HRV: {hrv:.1f}ms")
                
                components["hrv"] = {"value": hrv, "threshold": hrv_60th, "score": hrv_score}
                raw_score += hrv_score
        else:
            notes.append("No heart rate data available")
        
        return SubScoreDetail(
            agent_id=self.agent_id,
            name="heart_health_score",
            raw_score=raw_score,
            weight=weight,
            weighted_score=round(raw_score * weight, 2),
            max_points=max_points,
            components=components,
            notes=notes,
        )
    
    def _calculate_sleep_health_score(self, metrics: HealthMetrics) -> SubScoreDetail:
        """Calculate Sleep Health Score (15% weight, max 15 points)."""
        max_points = 15
        weight = self.SCORE_WEIGHTS["sleep_health"]
        components = {}
        notes = []
        raw_score = 0
        
        if metrics.sleep:
            # Sleep duration scoring (max 10 points)
            duration = metrics.sleep.avg_sleep_duration_hours
            if duration:
                if 7 <= duration <= 8:
                    duration_score = 10
                    notes.append(f"Optimal sleep: {duration:.1f} hours")
                elif 6 <= duration < 7 or 8 < duration <= 9:
                    duration_score = 6
                    notes.append(f"Good sleep: {duration:.1f} hours")
                elif duration > 9:
                    duration_score = 3
                    notes.append(f"Long sleep: {duration:.1f} hours")
                else:
                    duration_score = 0
                    notes.append(f"Short sleep: {duration:.1f} hours (target: 7-8)")
                
                components["duration"] = {"value": duration, "score": duration_score}
                raw_score += duration_score
            
            # Sleep consistency scoring (max 5 points)
            # Use sleep efficiency as proxy for consistency
            efficiency = metrics.sleep.sleep_efficiency_pct
            if efficiency:
                if efficiency >= 85:
                    consistency_score = 5
                    notes.append(f"Consistent sleep: {efficiency:.0f}% efficiency")
                elif efficiency >= 70:
                    consistency_score = 3
                    notes.append(f"Moderate sleep consistency: {efficiency:.0f}% efficiency")
                else:
                    consistency_score = 0
                    notes.append(f"Poor sleep consistency: {efficiency:.0f}% efficiency")
                
                components["consistency"] = {"value": efficiency, "score": consistency_score}
                raw_score += consistency_score
        else:
            notes.append("No sleep data available")
        
        return SubScoreDetail(
            agent_id=self.agent_id,
            name="sleep_health_score",
            raw_score=raw_score,
            weight=weight,
            weighted_score=round(raw_score * weight, 2),
            max_points=max_points,
            components=components,
            notes=notes,
        )
    
    def _calculate_body_composition_score(self, metrics: HealthMetrics, profile: PatientProfile) -> SubScoreDetail:
        """Calculate Body Composition Score (10% weight, max 10 points)."""
        max_points = 10
        weight = self.SCORE_WEIGHTS["body_composition"]
        components = {}
        notes = []
        raw_score = 0
        
        # Get BMI from profile if available
        bmi = None
        if profile.medical_history and profile.medical_history.bmi:
            bmi = profile.medical_history.bmi
        
        if bmi:
            # BMI-based scoring (stable = good)
            if 18.5 <= bmi <= 24.9:
                raw_score = 10
                notes.append(f"Healthy BMI: {bmi:.1f}")
            elif 25 <= bmi <= 29.9:
                raw_score = 5
                notes.append(f"Overweight BMI: {bmi:.1f}")
            elif bmi < 18.5:
                raw_score = 5
                notes.append(f"Underweight BMI: {bmi:.1f}")
            else:
                raw_score = 0
                notes.append(f"Obese BMI: {bmi:.1f}")
            
            components["bmi"] = {"value": bmi, "score": raw_score}
            
            # Check trend if available
            if metrics.trends and metrics.trends.overall_health_trajectory:
                trajectory = metrics.trends.overall_health_trajectory
                if trajectory == "positive":
                    notes.append("Body composition trend: Improving")
                elif trajectory == "negative":
                    notes.append("Body composition trend: Declining")
        else:
            notes.append("No BMI data available")
            raw_score = 5  # Default middle score
        
        return SubScoreDetail(
            agent_id=self.agent_id,
            name="body_composition_score",
            raw_score=raw_score,
            weight=weight,
            weighted_score=round(raw_score * weight, 2),
            max_points=max_points,
            components=components,
            notes=notes,
        )
    
    def _calculate_mobility_score(self, metrics: HealthMetrics, age: int) -> SubScoreDetail:
        """Calculate Mobility Score (10% weight, max 10 points)."""
        max_points = 10
        weight = self.SCORE_WEIGHTS["mobility"]
        components = {}
        notes = []
        raw_score = 0
        
        # Check for mobility metrics (may not be present in current schema)
        walking_speed = None
        walking_steadiness = None
        
        if hasattr(metrics, 'mobility') and metrics.mobility:
            walking_speed = getattr(metrics.mobility, 'walking_speed', None)
            walking_steadiness = getattr(metrics.mobility, 'walking_steadiness', None)
        
        if walking_speed:
            # Age-adjusted walking speed scoring (max 5 points)
            speed_threshold = self._get_walking_speed_60th_percentile(age)
            if walking_speed >= speed_threshold:
                speed_score = 5
                notes.append(f"Good walking speed: {walking_speed:.2f} m/s (≥60th percentile)")
            elif walking_speed >= speed_threshold * 0.75:
                speed_score = 3
                notes.append(f"Fair walking speed: {walking_speed:.2f} m/s")
            else:
                speed_score = 0
                notes.append(f"Below average walking speed: {walking_speed:.2f} m/s")
            
            components["walking_speed"] = {"value": walking_speed, "score": speed_score}
            raw_score += speed_score
        else:
            # Estimate from activity if no mobility data
            if metrics.activity and metrics.activity.daily_steps_avg:
                steps = metrics.activity.daily_steps_avg
                if steps > 8000:
                    raw_score += 5
                    notes.append("Walking speed: Estimated as good from activity level")
                elif steps > 5000:
                    raw_score += 3
                    notes.append("Walking speed: Estimated as fair from activity level")
                else:
                    notes.append("Walking speed: No data, estimated from low activity")
                components["estimated_from_activity"] = True
            else:
                notes.append("No walking speed data available")
        
        if walking_steadiness:
            # Steadiness scoring (max 5 points)
            if walking_steadiness == "normal":
                steadiness_score = 5
                notes.append("Normal walking steadiness")
            elif walking_steadiness == "slightly_unsteady":
                steadiness_score = 2
                notes.append("Slightly unsteady walking")
            else:
                steadiness_score = 0
                notes.append("Unsteady walking detected")
            
            components["steadiness"] = {"value": walking_steadiness, "score": steadiness_score}
            raw_score += steadiness_score
        else:
            # Default to stable if no issues reported
            raw_score += 5
            notes.append("Walking steadiness: Assumed normal (no issues detected)")
            components["steadiness_assumed"] = True
        
        return SubScoreDetail(
            agent_id=self.agent_id,
            name="mobility_score",
            raw_score=raw_score,
            weight=weight,
            weighted_score=round(raw_score * weight, 2),
            max_points=max_points,
            components=components,
            notes=notes,
        )
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _estimate_vo2_percentile(self, vo2_max: float, age: int) -> float:
        """Estimate VO2 max percentile based on age (simplified)."""
        # Simplified age-adjusted thresholds
        if age < 30:
            thresholds = [40, 45, 50]  # 25th, 50th, 75th
        elif age < 40:
            thresholds = [35, 40, 45]
        elif age < 50:
            thresholds = [30, 35, 40]
        elif age < 60:
            thresholds = [25, 30, 35]
        else:
            thresholds = [20, 25, 30]
        
        if vo2_max >= thresholds[2]:
            return 75
        elif vo2_max >= thresholds[1]:
            return 50
        elif vo2_max >= thresholds[0]:
            return 25
        else:
            return 10
    
    def _get_hrv_60th_percentile(self, age: int) -> float:
        """Get 60th percentile HRV threshold for age (simplified)."""
        # HRV generally decreases with age
        if age < 30:
            return 60
        elif age < 40:
            return 50
        elif age < 50:
            return 40
        elif age < 60:
            return 35
        else:
            return 30
    
    def _get_walking_speed_60th_percentile(self, age: int) -> float:
        """Get 60th percentile walking speed for age in m/s."""
        # Walking speed thresholds by age
        if age < 50:
            return 1.4
        elif age < 60:
            return 1.3
        elif age < 70:
            return 1.2
        else:
            return 1.0
    
    def _get_hkrs_band(self, hkrs: float) -> HKRSBand:
        """Determine HKRS band from score."""
        if hkrs >= 85:
            return HKRSBand.EXCELLENT
        elif hkrs >= 70:
            return HKRSBand.VERY_GOOD
        elif hkrs >= 55:
            return HKRSBand.STANDARD_PLUS
        elif hkrs >= 40:
            return HKRSBand.STANDARD
        else:
            return HKRSBand.SUBSTANDARD
    
    def _get_band_description(self, band: HKRSBand) -> str:
        """Get human-readable description for HKRS band."""
        descriptions = {
            HKRSBand.EXCELLENT: "Excellent health signals - eligible for best risk class",
            HKRSBand.VERY_GOOD: "Very good health signals - may improve risk class by one",
            HKRSBand.STANDARD_PLUS: "Standard Plus health signals",
            HKRSBand.STANDARD: "Standard health signals - no adjustment",
            HKRSBand.SUBSTANDARD: "Substandard / Manual Review Required",
        }
        return descriptions.get(band, "Unknown")
    
    def _get_risk_class_recommendation(
        self, 
        hkrs: float, 
        band: HKRSBand,
        data_quality: DataQuality
    ) -> Tuple[str, str]:
        """Determine risk class recommendation and action."""
        if data_quality == DataQuality.LOW:
            return "Manual Review Required", "insufficient_data_for_automation"
        
        if band == HKRSBand.EXCELLENT:
            return "Preferred Plus", "eligible_for_best_class_if_traditional_aligns"
        elif band == HKRSBand.VERY_GOOD:
            return "Preferred", "improve_one_class_max"
        elif band == HKRSBand.STANDARD_PLUS:
            return "Standard Plus", "standard_processing"
        elif band == HKRSBand.STANDARD:
            return "Standard", "no_positive_adjustment"
        else:
            return "Substandard", "manual_review_required"
    
    def _determine_decision(
        self, 
        band: HKRSBand, 
        data_quality: DataQuality
    ) -> Tuple[bool, str, bool]:
        """Determine approval status from HKRS band."""
        # HealthKit data is additive only - cannot auto-decline
        if data_quality == DataQuality.LOW:
            return True, "referred", True
        
        if band == HKRSBand.SUBSTANDARD:
            return True, "referred", True
        elif band in [HKRSBand.EXCELLENT, HKRSBand.VERY_GOOD]:
            return True, "approved", False
        else:
            return True, "approved_with_adjustment", False
    
    def _get_top_positive_drivers(self, sub_scores: List[SubScoreDetail]) -> List[str]:
        """Get top 3 positive factors from sub-scores."""
        # Sort by weighted score descending
        sorted_scores = sorted(sub_scores, key=lambda s: s.weighted_score, reverse=True)
        
        drivers = []
        for score in sorted_scores[:3]:
            if score.weighted_score > 0:
                # Extract the best note
                if score.notes:
                    drivers.append(score.notes[0])
        
        return drivers
    
    def _get_improvement_suggestions(
        self, 
        sub_scores: List[SubScoreDetail],
        metrics: HealthMetrics
    ) -> List[str]:
        """Generate improvement suggestions for low-scoring areas."""
        suggestions = []
        
        for score in sub_scores:
            # If scoring below 50% of max, suggest improvement
            if score.raw_score < score.max_points * 0.5:
                if score.name == "activity_score":
                    suggestions.append("Increase daily steps to 8,000+ for better activity score")
                elif score.name == "sleep_health_score":
                    suggestions.append("Aim for 7-8 hours of consistent sleep each night")
                elif score.name == "heart_health_score":
                    suggestions.append("Regular cardio exercise can improve heart rate metrics")
                elif score.name == "body_composition_score":
                    suggestions.append("Maintain a healthy weight through balanced diet and exercise")
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def _generate_summary_scorecard(
        self, 
        hkrs: float, 
        band: HKRSBand,
        top_drivers: List[str]
    ) -> str:
        """Generate applicant-facing summary scorecard."""
        summary = f"Your HealthKit Risk Score is {hkrs:.0f}/100 ({band.value.replace('_', ' ').title()})."
        
        if top_drivers:
            summary += f" {top_drivers[0]}"
        
        return summary
    
    def _generate_rationale(
        self, 
        hkrs: float, 
        band: HKRSBand,
        sub_scores: List[SubScoreDetail],
        data_quality: DataQuality
    ) -> str:
        """Generate detailed rationale for the decision."""
        parts = [f"HealthKit Risk Score: {hkrs:.1f}/100 ({band.value.replace('_', ' ').title()})"]
        
        parts.append(f"Data Quality: {data_quality.value.title()}")
        
        # Add sub-score summary
        for score in sub_scores:
            parts.append(f"- {score.name.replace('_', ' ').title()}: {score.raw_score}/{score.max_points}")
        
        return "; ".join(parts)
