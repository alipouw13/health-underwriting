"""
Mock Apple Health Data Generator

SIMULATED DATA - NOT FOR PRODUCTION USE
This module generates synthetic Apple Health data for demo purposes.
NO real Apple APIs, OAuth, or HealthKit SDKs are used.

The mock data is designed to produce DIFFERENT risk analysis results
based on user profiles, demonstrating the agent pipeline's ability
to assess varying health conditions.
"""

import hashlib
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from enum import Enum


class HealthProfile(str, Enum):
    """Predefined health profiles for mock data generation."""
    EXCELLENT = "excellent"  # Very healthy, low risk
    GOOD = "good"           # Generally healthy, minor concerns
    MODERATE = "moderate"   # Some health concerns
    CONCERNING = "concerning"  # Multiple risk factors
    HIGH_RISK = "high_risk"   # Significant health issues


@dataclass
class AppleHealthMockData:
    """
    Mock Apple Health data structure.
    
    This mirrors what would be returned from a real Apple HealthKit integration.
    Data is marked as source="apple_health_mock" to distinguish from real data.
    """
    user_id: str
    consent_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data_source: str = "apple_health_mock"
    
    # Activity Data
    daily_steps_avg: int = 8000
    daily_active_minutes_avg: int = 45
    daily_calories_burned_avg: int = 2200
    weekly_exercise_sessions: int = 3
    activity_days_with_data: int = 85
    
    # Heart Rate Data
    resting_hr_avg: int = 68
    resting_hr_min: int = 58
    resting_hr_max: int = 82
    hrv_avg_ms: float = 42.0
    elevated_hr_events: int = 2
    irregular_rhythm_events: int = 0
    heart_rate_days_with_data: int = 88
    
    # Sleep Data
    avg_sleep_duration_hours: float = 7.2
    avg_time_to_sleep_minutes: int = 15
    sleep_efficiency_pct: float = 88.0
    deep_sleep_pct: float = 18.0
    rem_sleep_pct: float = 22.0
    light_sleep_pct: float = 60.0
    avg_awakenings_per_night: float = 1.5
    sleep_nights_with_data: int = 82
    
    # Biometrics
    height_cm: float = 175.0
    weight_kg: float = 75.0
    bmi: float = 24.5
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    
    # Trends
    activity_trend_weekly: str = "stable"
    resting_hr_trend_weekly: str = "stable"
    sleep_quality_trend_weekly: str = "stable"
    overall_health_trajectory: str = "neutral"
    significant_changes: List[str] = field(default_factory=list)
    
    # Health Flags (derived from data)
    has_elevated_hr_concern: bool = False
    has_irregular_rhythm_concern: bool = False
    has_sleep_concern: bool = False
    has_activity_concern: bool = False
    has_bmi_concern: bool = False
    
    # Measurement period
    measurement_period_days: int = 90
    last_recorded_date: date = field(default_factory=date.today)
    
    def to_health_metrics_dict(self) -> Dict[str, Any]:
        """Convert to the HealthMetrics schema format used by agents."""
        return {
            "patient_id": self.user_id,
            "data_source": self.data_source,
            "collection_timestamp": self.consent_timestamp.isoformat(),
            "activity": {
                "daily_steps_avg": self.daily_steps_avg,
                "daily_active_minutes_avg": self.daily_active_minutes_avg,
                "daily_calories_burned_avg": self.daily_calories_burned_avg,
                "weekly_exercise_sessions": self.weekly_exercise_sessions,
                "days_with_data": self.activity_days_with_data,
                "measurement_period_days": self.measurement_period_days,
                "last_recorded_date": self.last_recorded_date.isoformat(),
            },
            "heart_rate": {
                "resting_hr_avg": self.resting_hr_avg,
                "resting_hr_min": self.resting_hr_min,
                "resting_hr_max": self.resting_hr_max,
                "hrv_avg_ms": self.hrv_avg_ms,
                "elevated_hr_events": self.elevated_hr_events,
                "irregular_rhythm_events": self.irregular_rhythm_events,
                "days_with_data": self.heart_rate_days_with_data,
                "measurement_period_days": self.measurement_period_days,
                "last_recorded_date": self.last_recorded_date.isoformat(),
            },
            "sleep": {
                "avg_sleep_duration_hours": self.avg_sleep_duration_hours,
                "avg_time_to_sleep_minutes": self.avg_time_to_sleep_minutes,
                "sleep_efficiency_pct": self.sleep_efficiency_pct,
                "deep_sleep_pct": self.deep_sleep_pct,
                "rem_sleep_pct": self.rem_sleep_pct,
                "light_sleep_pct": self.light_sleep_pct,
                "avg_awakenings_per_night": self.avg_awakenings_per_night,
                "nights_with_data": self.sleep_nights_with_data,
                "measurement_period_days": self.measurement_period_days,
                "last_recorded_date": self.last_recorded_date.isoformat(),
            },
            "trends": {
                "activity_trend_weekly": self.activity_trend_weekly,
                "activity_trend_monthly": self.activity_trend_weekly,  # Simplified
                "resting_hr_trend_weekly": self.resting_hr_trend_weekly,
                "resting_hr_trend_monthly": self.resting_hr_trend_weekly,
                "sleep_quality_trend_weekly": self.sleep_quality_trend_weekly,
                "sleep_quality_trend_monthly": self.sleep_quality_trend_weekly,
                "overall_health_trajectory": self.overall_health_trajectory,
                "significant_changes": self.significant_changes,
            },
            "consent_verified": True,
            "data_anonymized": True,
        }
    
    def to_patient_profile_dict(
        self,
        first_name: str,
        last_name: str,
        date_of_birth: date,
        biological_sex: str = "unknown",
        policy_type: str = "term_life",
        coverage_amount: float = 500000.0,
    ) -> Dict[str, Any]:
        """Convert to the PatientProfile schema format used by agents."""
        age = (date.today() - date_of_birth).days // 365
        
        # Derive medical history from mock data
        has_hypertension = self.blood_pressure_systolic is not None and self.blood_pressure_systolic >= 140
        has_heart_disease = self.irregular_rhythm_events > 5 or self.elevated_hr_events > 20
        
        # Determine smoker status based on health indicators
        # (Mock logic: poor sleep + low activity might correlate with smoking)
        smoker_status = "never"
        if self.avg_sleep_duration_hours < 5.5 and self.daily_active_minutes_avg < 20:
            smoker_status = "former"  # Conservative assumption
        
        return {
            "patient_id": self.user_id,
            "demographics": {
                "age": age,
                "biological_sex": biological_sex,
                "state_region": "California",  # Default for demo
            },
            "medical_history": {
                "has_diabetes": False,  # Would need additional data
                "has_hypertension": has_hypertension,
                "has_heart_disease": has_heart_disease,
                "has_cancer_history": False,
                "smoker_status": smoker_status,
                "alcohol_use": "moderate",  # Default
                "bmi": self.bmi,
                "family_history_heart_disease": None,
                "family_history_cancer": None,
                "family_history_diabetes": None,
            },
            "policy_type_requested": policy_type,
            "coverage_amount_requested": coverage_amount,
            "profile_created_date": date.today().isoformat(),
        }


def _get_profile_from_seed(seed: int) -> HealthProfile:
    """Deterministically select a health profile based on seed."""
    # Use modulo to distribute profiles
    profiles = [
        HealthProfile.EXCELLENT,
        HealthProfile.GOOD,
        HealthProfile.GOOD,
        HealthProfile.MODERATE,
        HealthProfile.MODERATE,
        HealthProfile.MODERATE,
        HealthProfile.CONCERNING,
        HealthProfile.CONCERNING,
        HealthProfile.HIGH_RISK,
        HealthProfile.HIGH_RISK,
    ]
    return profiles[seed % len(profiles)]


def generate_apple_health_data(
    user_id: str,
    date_of_birth: date,
    profile_override: Optional[HealthProfile] = None,
    randomize: bool = True,
) -> AppleHealthMockData:
    """
    Generate mock Apple Health data based on user_id and DOB.
    
    By default (randomize=True), generates RANDOM data each time.
    Set randomize=False for deterministic testing.
    
    Args:
        user_id: Unique user identifier
        date_of_birth: User's date of birth (affects age-related metrics)
        profile_override: Optional override for the health profile
        randomize: If True, generate random data; if False, use deterministic seed
        
    Returns:
        AppleHealthMockData object with synthetic health data
    """
    if randomize:
        # Truly random data each time
        rng = random.Random()
        # Randomly select a health profile with weighted distribution
        profiles_weighted = [
            HealthProfile.EXCELLENT,
            HealthProfile.GOOD,
            HealthProfile.GOOD,
            HealthProfile.MODERATE,
            HealthProfile.MODERATE,
            HealthProfile.MODERATE,
            HealthProfile.CONCERNING,
            HealthProfile.HIGH_RISK,
        ]
        profile = profile_override or rng.choice(profiles_weighted)
    else:
        # Create deterministic seed from user_id
        hash_bytes = hashlib.sha256(user_id.encode()).digest()
        seed = int.from_bytes(hash_bytes[:4], 'big')
        rng = random.Random(seed)
        # Determine health profile
        profile = profile_override or _get_profile_from_seed(seed)
    
    # Calculate age for age-appropriate metrics
    age = (date.today() - date_of_birth).days // 365
    
    # Generate base metrics based on profile
    if profile == HealthProfile.EXCELLENT:
        data = _generate_excellent_profile(rng, age)
    elif profile == HealthProfile.GOOD:
        data = _generate_good_profile(rng, age)
    elif profile == HealthProfile.MODERATE:
        data = _generate_moderate_profile(rng, age)
    elif profile == HealthProfile.CONCERNING:
        data = _generate_concerning_profile(rng, age)
    else:  # HIGH_RISK
        data = _generate_high_risk_profile(rng, age)
    
    data.user_id = user_id
    data.last_recorded_date = date.today() - timedelta(days=rng.randint(0, 3))
    
    # Set health flags based on data
    data.has_elevated_hr_concern = data.elevated_hr_events > 5
    data.has_irregular_rhythm_concern = data.irregular_rhythm_events > 0
    data.has_sleep_concern = data.avg_sleep_duration_hours < 6.0 or data.sleep_efficiency_pct < 75
    data.has_activity_concern = data.daily_steps_avg < 5000 or data.daily_active_minutes_avg < 20
    data.has_bmi_concern = data.bmi > 30 or data.bmi < 18.5
    
    return data


def _generate_excellent_profile(rng: random.Random, age: int) -> AppleHealthMockData:
    """Generate data for an excellent health profile."""
    return AppleHealthMockData(
        user_id="",  # Set by caller
        # Activity - very active
        daily_steps_avg=rng.randint(10000, 14000),
        daily_active_minutes_avg=rng.randint(60, 90),
        daily_calories_burned_avg=rng.randint(2400, 3000),
        weekly_exercise_sessions=rng.randint(4, 6),
        activity_days_with_data=rng.randint(88, 90),
        # Heart Rate - excellent
        resting_hr_avg=rng.randint(55, 62),
        resting_hr_min=rng.randint(48, 55),
        resting_hr_max=rng.randint(65, 72),
        hrv_avg_ms=rng.uniform(50, 70),
        elevated_hr_events=rng.randint(0, 1),
        irregular_rhythm_events=0,
        heart_rate_days_with_data=rng.randint(88, 90),
        # Sleep - excellent
        avg_sleep_duration_hours=rng.uniform(7.5, 8.5),
        avg_time_to_sleep_minutes=rng.randint(5, 12),
        sleep_efficiency_pct=rng.uniform(92, 97),
        deep_sleep_pct=rng.uniform(20, 25),
        rem_sleep_pct=rng.uniform(22, 28),
        light_sleep_pct=rng.uniform(50, 55),
        avg_awakenings_per_night=rng.uniform(0.5, 1.2),
        sleep_nights_with_data=rng.randint(85, 90),
        # Biometrics - healthy
        height_cm=rng.uniform(165, 185),
        weight_kg=rng.uniform(60, 80),
        bmi=rng.uniform(20, 24),
        blood_pressure_systolic=rng.randint(105, 118),
        blood_pressure_diastolic=rng.randint(65, 75),
        # Trends - positive
        activity_trend_weekly="improving",
        resting_hr_trend_weekly="improving",
        sleep_quality_trend_weekly="stable",
        overall_health_trajectory="positive",
        significant_changes=[],
    )


def _generate_good_profile(rng: random.Random, age: int) -> AppleHealthMockData:
    """Generate data for a good health profile."""
    return AppleHealthMockData(
        user_id="",
        # Activity - moderately active
        daily_steps_avg=rng.randint(7500, 10000),
        daily_active_minutes_avg=rng.randint(40, 60),
        daily_calories_burned_avg=rng.randint(2100, 2500),
        weekly_exercise_sessions=rng.randint(3, 4),
        activity_days_with_data=rng.randint(82, 88),
        # Heart Rate - good
        resting_hr_avg=rng.randint(62, 70),
        resting_hr_min=rng.randint(55, 62),
        resting_hr_max=rng.randint(72, 82),
        hrv_avg_ms=rng.uniform(38, 52),
        elevated_hr_events=rng.randint(1, 4),
        irregular_rhythm_events=0,
        heart_rate_days_with_data=rng.randint(85, 90),
        # Sleep - good
        avg_sleep_duration_hours=rng.uniform(6.8, 7.5),
        avg_time_to_sleep_minutes=rng.randint(10, 20),
        sleep_efficiency_pct=rng.uniform(85, 92),
        deep_sleep_pct=rng.uniform(16, 20),
        rem_sleep_pct=rng.uniform(20, 24),
        light_sleep_pct=rng.uniform(58, 62),
        avg_awakenings_per_night=rng.uniform(1.0, 2.0),
        sleep_nights_with_data=rng.randint(80, 88),
        # Biometrics - healthy
        height_cm=rng.uniform(165, 185),
        weight_kg=rng.uniform(65, 85),
        bmi=rng.uniform(22, 26),
        blood_pressure_systolic=rng.randint(115, 125),
        blood_pressure_diastolic=rng.randint(72, 82),
        # Trends - stable
        activity_trend_weekly="stable",
        resting_hr_trend_weekly="stable",
        sleep_quality_trend_weekly="stable",
        overall_health_trajectory="neutral",
        significant_changes=[],
    )


def _generate_moderate_profile(rng: random.Random, age: int) -> AppleHealthMockData:
    """Generate data for a moderate health profile with some concerns."""
    return AppleHealthMockData(
        user_id="",
        # Activity - somewhat active
        daily_steps_avg=rng.randint(5500, 8000),
        daily_active_minutes_avg=rng.randint(25, 45),
        daily_calories_burned_avg=rng.randint(1900, 2200),
        weekly_exercise_sessions=rng.randint(2, 3),
        activity_days_with_data=rng.randint(70, 82),
        # Heart Rate - moderate
        resting_hr_avg=rng.randint(70, 78),
        resting_hr_min=rng.randint(62, 70),
        resting_hr_max=rng.randint(82, 92),
        hrv_avg_ms=rng.uniform(28, 42),
        elevated_hr_events=rng.randint(5, 12),
        irregular_rhythm_events=rng.randint(0, 1),
        heart_rate_days_with_data=rng.randint(80, 88),
        # Sleep - some issues
        avg_sleep_duration_hours=rng.uniform(6.0, 7.0),
        avg_time_to_sleep_minutes=rng.randint(20, 35),
        sleep_efficiency_pct=rng.uniform(78, 86),
        deep_sleep_pct=rng.uniform(12, 17),
        rem_sleep_pct=rng.uniform(18, 22),
        light_sleep_pct=rng.uniform(62, 68),
        avg_awakenings_per_night=rng.uniform(2.0, 3.5),
        sleep_nights_with_data=rng.randint(72, 82),
        # Biometrics - borderline
        height_cm=rng.uniform(165, 185),
        weight_kg=rng.uniform(75, 95),
        bmi=rng.uniform(26, 29),
        blood_pressure_systolic=rng.randint(125, 135),
        blood_pressure_diastolic=rng.randint(80, 88),
        # Trends - mixed
        activity_trend_weekly="declining",
        resting_hr_trend_weekly="stable",
        sleep_quality_trend_weekly="declining",
        overall_health_trajectory="neutral",
        significant_changes=["Activity levels decreased over past month"],
    )


def _generate_concerning_profile(rng: random.Random, age: int) -> AppleHealthMockData:
    """Generate data for a concerning health profile."""
    return AppleHealthMockData(
        user_id="",
        # Activity - sedentary
        daily_steps_avg=rng.randint(3500, 5500),
        daily_active_minutes_avg=rng.randint(15, 25),
        daily_calories_burned_avg=rng.randint(1700, 2000),
        weekly_exercise_sessions=rng.randint(0, 2),
        activity_days_with_data=rng.randint(60, 75),
        # Heart Rate - elevated
        resting_hr_avg=rng.randint(78, 88),
        resting_hr_min=rng.randint(70, 78),
        resting_hr_max=rng.randint(92, 105),
        hrv_avg_ms=rng.uniform(20, 30),
        elevated_hr_events=rng.randint(15, 30),
        irregular_rhythm_events=rng.randint(1, 4),
        heart_rate_days_with_data=rng.randint(75, 85),
        # Sleep - poor
        avg_sleep_duration_hours=rng.uniform(5.2, 6.2),
        avg_time_to_sleep_minutes=rng.randint(35, 55),
        sleep_efficiency_pct=rng.uniform(70, 78),
        deep_sleep_pct=rng.uniform(8, 13),
        rem_sleep_pct=rng.uniform(14, 19),
        light_sleep_pct=rng.uniform(68, 75),
        avg_awakenings_per_night=rng.uniform(3.5, 5.5),
        sleep_nights_with_data=rng.randint(65, 78),
        # Biometrics - overweight
        height_cm=rng.uniform(165, 185),
        weight_kg=rng.uniform(88, 110),
        bmi=rng.uniform(29, 33),
        blood_pressure_systolic=rng.randint(135, 148),
        blood_pressure_diastolic=rng.randint(88, 95),
        # Trends - negative
        activity_trend_weekly="declining",
        resting_hr_trend_weekly="concerning",
        sleep_quality_trend_weekly="declining",
        overall_health_trajectory="negative",
        significant_changes=[
            "Resting heart rate increased by 8 bpm over past 2 months",
            "Sleep duration decreased significantly",
            "Multiple irregular rhythm events detected",
        ],
    )


def _generate_high_risk_profile(rng: random.Random, age: int) -> AppleHealthMockData:
    """Generate data for a high risk health profile."""
    return AppleHealthMockData(
        user_id="",
        # Activity - very sedentary
        daily_steps_avg=rng.randint(2000, 4000),
        daily_active_minutes_avg=rng.randint(5, 18),
        daily_calories_burned_avg=rng.randint(1500, 1800),
        weekly_exercise_sessions=0,
        activity_days_with_data=rng.randint(45, 65),
        # Heart Rate - concerning
        resting_hr_avg=rng.randint(88, 100),
        resting_hr_min=rng.randint(78, 88),
        resting_hr_max=rng.randint(105, 125),
        hrv_avg_ms=rng.uniform(12, 22),
        elevated_hr_events=rng.randint(30, 60),
        irregular_rhythm_events=rng.randint(5, 15),
        heart_rate_days_with_data=rng.randint(65, 80),
        # Sleep - very poor
        avg_sleep_duration_hours=rng.uniform(4.5, 5.5),
        avg_time_to_sleep_minutes=rng.randint(50, 90),
        sleep_efficiency_pct=rng.uniform(60, 72),
        deep_sleep_pct=rng.uniform(5, 10),
        rem_sleep_pct=rng.uniform(10, 15),
        light_sleep_pct=rng.uniform(75, 82),
        avg_awakenings_per_night=rng.uniform(5.0, 8.0),
        sleep_nights_with_data=rng.randint(50, 68),
        # Biometrics - obese
        height_cm=rng.uniform(165, 185),
        weight_kg=rng.uniform(105, 140),
        bmi=rng.uniform(33, 42),
        blood_pressure_systolic=rng.randint(148, 165),
        blood_pressure_diastolic=rng.randint(95, 108),
        # Trends - very negative
        activity_trend_weekly="declining",
        resting_hr_trend_weekly="concerning",
        sleep_quality_trend_weekly="declining",
        overall_health_trajectory="negative",
        significant_changes=[
            "Resting heart rate consistently elevated above 90 bpm",
            "Multiple irregular rhythm events detected - medical review recommended",
            "Severe sleep deficiency detected",
            "Very low activity levels - virtually sedentary",
            "BMI indicates obesity - increased cardiovascular risk",
        ],
    )
