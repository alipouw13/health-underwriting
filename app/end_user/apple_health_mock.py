"""
Mock Apple Health Data Generator

SIMULATED DATA - NOT FOR PRODUCTION USE
This module generates synthetic Apple Health data for demo purposes.
NO real Apple APIs, OAuth, or HealthKit SDKs are used.

The mock data aligns with the Apple HealthKit Underwriting Rules specification:
- Activity: Steps/day, Active Energy
- Fitness: VO2 Max
- Vitals: Resting HR, HRV  
- Sleep: Duration, consistency
- Body Metrics: Weight trend, BMI trend
- Mobility: Walking speed, steadiness
- Exercise: Workout frequency & intensity

Data Categories explicitly excluded per policy:
- Reproductive health
- Mental health notes
- GPS / location data
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
    Mock Apple Health data structure aligned with HealthKit Underwriting Rules.
    
    Categories from Apple UW Manual:
    1. Activity - Steps/day, Active Energy
    2. Fitness - VO2 Max (cardiorespiratory fitness)
    3. Vitals - Resting HR, HRV
    4. Sleep - Duration, consistency
    5. Body Metrics - Weight trend, BMI
    6. Mobility - Walking speed, steadiness
    7. Exercise - Workout frequency & intensity
    
    Data is marked as source="apple_health_mock" to distinguish from real data.
    """
    user_id: str
    consent_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data_source: str = "apple_health_mock"
    
    # =========================================================================
    # 1. ACTIVITY DATA (25% weight in HKRS)
    # =========================================================================
    daily_steps_avg: int = 8000
    active_energy_burned_avg: float = 400.0  # kcal/day from active movement
    activity_days_with_data: int = 120  # Min 120 days per policy
    
    # =========================================================================
    # 2. FITNESS DATA (20% weight in HKRS)
    # =========================================================================
    vo2_max: Optional[float] = 38.0  # mL/kg/min - cardiorespiratory fitness
    vo2_max_readings: int = 5  # Min 3 readings per policy
    
    # =========================================================================
    # 3. VITALS DATA (20% weight in HKRS)
    # =========================================================================
    resting_hr_avg: int = 68  # bpm - preferred 50-70
    hrv_avg_ms: float = 42.0  # Heart Rate Variability in ms
    elevated_hr_events: int = 2  # Count of elevated HR at rest
    irregular_rhythm_events: int = 0  # AFib/arrhythmia events
    heart_rate_days_with_data: int = 90  # Min 60 days per policy
    
    # =========================================================================
    # 4. SLEEP DATA (15% weight in HKRS)
    # =========================================================================
    avg_sleep_duration_hours: float = 7.2  # Optimal: 7-8 hours
    sleep_consistency_variance_hours: float = 0.8  # +/- variance, optimal: <1 hour
    sleep_nights_with_data: int = 100  # Min 90 days per policy
    
    # =========================================================================
    # 5. BODY METRICS DATA (10% weight in HKRS)
    # =========================================================================
    bmi: float = 24.5
    bmi_trend: str = "stable"  # stable, improving, mild_increase, significant_increase
    weight_kg: float = 75.0
    height_cm: float = 175.0
    
    # =========================================================================
    # 6. MOBILITY DATA (10% weight in HKRS)
    # =========================================================================
    walking_speed_avg: float = 1.3  # m/s - 60th percentile threshold varies by age
    walking_steadiness: str = "normal"  # normal, slightly_unsteady, unsteady
    double_support_time_pct: Optional[float] = 25.0  # % of gait cycle
    
    # =========================================================================
    # 7. EXERCISE DATA (Part of Activity score)
    # =========================================================================
    workout_frequency_weekly: int = 3  # Sessions per week
    workout_avg_duration_minutes: int = 45
    workout_types: List[str] = field(default_factory=lambda: ["walking", "running"])
    
    # =========================================================================
    # TRENDS (used for scoring adjustments)
    # =========================================================================
    activity_trend_6mo: str = "stable"  # improving, stable, declining
    weight_trend_6mo: str = "stable"  # stable_or_improving, mild_increase, significant_increase
    overall_health_trajectory: str = "neutral"  # positive, neutral, negative
    significant_changes: List[str] = field(default_factory=list)
    
    # =========================================================================
    # DATA QUALITY METRICS
    # =========================================================================
    measurement_period_days: int = 365  # 12 month lookback
    last_recorded_date: date = field(default_factory=date.today)
    
    # Health Concern Flags (derived from data)
    has_elevated_hr_concern: bool = False
    has_irregular_rhythm_concern: bool = False
    has_sleep_concern: bool = False
    has_activity_concern: bool = False
    has_bmi_concern: bool = False
    
    def to_health_metrics_dict(self) -> Dict[str, Any]:
        """Convert to the HealthMetrics schema format used by agents.
        
        Aligns with Apple HealthKit Underwriting Rules categories:
        - Activity, Fitness, Vitals, Sleep, Body Metrics, Mobility, Exercise
        """
        return {
            "patient_id": self.user_id,
            "data_source": self.data_source,
            "collection_timestamp": self.consent_timestamp.isoformat(),
            
            # 1. Activity (25% weight)
            "activity": {
                "daily_steps_avg": self.daily_steps_avg,
                "active_energy_burned_avg": self.active_energy_burned_avg,
                "days_with_data": self.activity_days_with_data,
                "measurement_period_days": self.measurement_period_days,
                "last_recorded_date": self.last_recorded_date.isoformat(),
                "trend_6mo": self.activity_trend_6mo,
            },
            
            # 2. Fitness (20% weight) - NEW
            "fitness": {
                "vo2_max": self.vo2_max,
                "vo2_max_readings": self.vo2_max_readings,
                "measurement_period_days": self.measurement_period_days,
            },
            
            # 3. Vitals / Heart Rate (20% weight)
            "heart_rate": {
                "resting_hr_avg": self.resting_hr_avg,
                "hrv_avg_ms": self.hrv_avg_ms,
                "elevated_hr_events": self.elevated_hr_events,
                "irregular_rhythm_events": self.irregular_rhythm_events,
                "days_with_data": self.heart_rate_days_with_data,
                "measurement_period_days": self.measurement_period_days,
                "last_recorded_date": self.last_recorded_date.isoformat(),
            },
            
            # 4. Sleep (15% weight)
            "sleep": {
                "avg_sleep_duration_hours": self.avg_sleep_duration_hours,
                "sleep_consistency_variance_hours": self.sleep_consistency_variance_hours,
                "nights_with_data": self.sleep_nights_with_data,
                "measurement_period_days": self.measurement_period_days,
                "last_recorded_date": self.last_recorded_date.isoformat(),
            },
            
            # 5. Body Metrics (10% weight) - NEW category name
            "body_metrics": {
                "bmi": self.bmi,
                "bmi_trend": self.bmi_trend,
                "weight_kg": self.weight_kg,
                "height_cm": self.height_cm,
                "weight_trend_6mo": self.weight_trend_6mo,
            },
            
            # 6. Mobility (10% weight) - NEW
            "mobility": {
                "walking_speed_avg": self.walking_speed_avg,
                "walking_steadiness": self.walking_steadiness,
                "double_support_time_pct": self.double_support_time_pct,
            },
            
            # 7. Exercise (part of Activity score)
            "exercise": {
                "workout_frequency_weekly": self.workout_frequency_weekly,
                "workout_avg_duration_minutes": self.workout_avg_duration_minutes,
                "workout_types": self.workout_types,
            },
            
            # Trends for HKRS calculation
            "trends": {
                "activity_trend_6mo": self.activity_trend_6mo,
                "weight_trend_6mo": self.weight_trend_6mo,
                "overall_health_trajectory": self.overall_health_trajectory,
                "significant_changes": self.significant_changes,
            },
            
            # Consent and privacy
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
        """Convert to the PatientProfile schema format used by agents.
        
        Only includes manual input fields + derived health flags.
        """
        age = (date.today() - date_of_birth).days // 365
        
        return {
            "patient_id": self.user_id,
            "demographics": {
                "age": age,
                "biological_sex": biological_sex,
                "first_name": first_name,
                "last_name": last_name,
                "date_of_birth": date_of_birth.isoformat(),
            },
            "medical_history": {
                "bmi": self.bmi,
                "height_cm": self.height_cm,
                "weight_kg": self.weight_kg,
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
    
    # Set health flags based on data (per Apple UW manual thresholds)
    data.has_elevated_hr_concern = data.resting_hr_avg > 80
    data.has_irregular_rhythm_concern = data.irregular_rhythm_events > 0
    data.has_sleep_concern = data.avg_sleep_duration_hours < 6.0
    data.has_activity_concern = data.daily_steps_avg < 4000
    data.has_bmi_concern = data.bmi > 30 or data.bmi < 18.5
    
    return data


def _generate_excellent_profile(rng: random.Random, age: int) -> AppleHealthMockData:
    """Generate data for an excellent health profile - HKRS 85-100."""
    return AppleHealthMockData(
        user_id="",  # Set by caller
        
        # 1. Activity (25%) - Very active, 25/25 points
        daily_steps_avg=rng.randint(10000, 14000),
        active_energy_burned_avg=rng.uniform(450, 600),
        activity_days_with_data=rng.randint(130, 150),
        
        # 2. Fitness (20%) - Excellent VO2, 20/20 points
        vo2_max=rng.uniform(45, 55),  # >75th percentile
        vo2_max_readings=rng.randint(8, 12),
        
        # 3. Vitals (20%) - Optimal HR/HRV, 20/20 points
        resting_hr_avg=rng.randint(52, 65),  # 50-70 optimal
        hrv_avg_ms=rng.uniform(55, 75),  # >60th percentile
        elevated_hr_events=0,
        irregular_rhythm_events=0,
        heart_rate_days_with_data=rng.randint(90, 120),
        
        # 4. Sleep (15%) - Optimal duration/consistency, 15/15 points
        avg_sleep_duration_hours=rng.uniform(7.2, 7.8),
        sleep_consistency_variance_hours=rng.uniform(0.3, 0.8),
        sleep_nights_with_data=rng.randint(100, 120),
        
        # 5. Body Metrics (10%) - Healthy BMI, stable, 10/10 points
        bmi=rng.uniform(20, 24),
        bmi_trend="stable",
        weight_kg=rng.uniform(60, 80),
        height_cm=rng.uniform(165, 185),
        
        # 6. Mobility (10%) - Excellent, 10/10 points
        walking_speed_avg=rng.uniform(1.4, 1.6),  # >60th percentile
        walking_steadiness="normal",
        double_support_time_pct=rng.uniform(22, 26),
        
        # 7. Exercise
        workout_frequency_weekly=rng.randint(4, 6),
        workout_avg_duration_minutes=rng.randint(45, 75),
        workout_types=["running", "cycling", "strength_training"],
        
        # Trends - positive
        activity_trend_6mo="improving",
        weight_trend_6mo="stable_or_improving",
        overall_health_trajectory="positive",
        significant_changes=[],
    )


def _generate_good_profile(rng: random.Random, age: int) -> AppleHealthMockData:
    """Generate data for a good health profile - HKRS 70-84."""
    return AppleHealthMockData(
        user_id="",
        
        # 1. Activity (25%) - Good, 18-25/25 points
        daily_steps_avg=rng.randint(7000, 9500),
        active_energy_burned_avg=rng.uniform(350, 450),
        activity_days_with_data=rng.randint(110, 130),
        
        # 2. Fitness (20%) - Good VO2, 15/20 points
        vo2_max=rng.uniform(38, 45),  # 50-74th percentile
        vo2_max_readings=rng.randint(5, 8),
        
        # 3. Vitals (20%) - Good HR/HRV, 15-20/20 points
        resting_hr_avg=rng.randint(60, 72),
        hrv_avg_ms=rng.uniform(42, 55),
        elevated_hr_events=rng.randint(0, 3),
        irregular_rhythm_events=0,
        heart_rate_days_with_data=rng.randint(80, 100),
        
        # 4. Sleep (15%) - Good, 10-15/15 points
        avg_sleep_duration_hours=rng.uniform(6.8, 7.5),
        sleep_consistency_variance_hours=rng.uniform(0.6, 1.2),
        sleep_nights_with_data=rng.randint(90, 110),
        
        # 5. Body Metrics (10%) - Good, 8-10/10 points
        bmi=rng.uniform(22, 26),
        bmi_trend="stable",
        weight_kg=rng.uniform(65, 85),
        height_cm=rng.uniform(165, 185),
        
        # 6. Mobility (10%) - Good, 8-10/10 points
        walking_speed_avg=rng.uniform(1.25, 1.4),
        walking_steadiness="normal",
        double_support_time_pct=rng.uniform(24, 28),
        
        # 7. Exercise
        workout_frequency_weekly=rng.randint(2, 4),
        workout_avg_duration_minutes=rng.randint(30, 50),
        workout_types=["walking", "yoga"],
        
        # Trends - stable
        activity_trend_6mo="stable",
        weight_trend_6mo="stable_or_improving",
        overall_health_trajectory="neutral",
        significant_changes=[],
    )


def _generate_moderate_profile(rng: random.Random, age: int) -> AppleHealthMockData:
    """Generate data for a moderate health profile - HKRS 55-69."""
    return AppleHealthMockData(
        user_id="",
        
        # 1. Activity (25%) - Moderate, 10-18/25 points
        daily_steps_avg=rng.randint(5000, 7000),
        active_energy_burned_avg=rng.uniform(250, 350),
        activity_days_with_data=rng.randint(90, 115),
        
        # 2. Fitness (20%) - Fair VO2, 8-15/20 points
        vo2_max=rng.uniform(30, 38),  # 25-49th percentile
        vo2_max_readings=rng.randint(3, 6),
        
        # 3. Vitals (20%) - Elevated HR, 10-15/20 points
        resting_hr_avg=rng.randint(72, 78),
        hrv_avg_ms=rng.uniform(32, 42),
        elevated_hr_events=rng.randint(3, 8),
        irregular_rhythm_events=rng.randint(0, 1),
        heart_rate_days_with_data=rng.randint(70, 90),
        
        # 4. Sleep (15%) - Some issues, 6-10/15 points
        avg_sleep_duration_hours=rng.uniform(6.0, 6.8),
        sleep_consistency_variance_hours=rng.uniform(1.2, 1.8),
        sleep_nights_with_data=rng.randint(75, 95),
        
        # 5. Body Metrics (10%) - Borderline, 5/10 points
        bmi=rng.uniform(26, 29),
        bmi_trend="mild_increase",
        weight_kg=rng.uniform(80, 95),
        height_cm=rng.uniform(165, 185),
        
        # 6. Mobility (10%) - Fair, 5-8/10 points
        walking_speed_avg=rng.uniform(1.1, 1.25),
        walking_steadiness="normal",
        double_support_time_pct=rng.uniform(27, 31),
        
        # 7. Exercise
        workout_frequency_weekly=rng.randint(1, 2),
        workout_avg_duration_minutes=rng.randint(20, 35),
        workout_types=["walking"],
        
        # Trends - declining
        activity_trend_6mo="declining",
        weight_trend_6mo="mild_increase",
        overall_health_trajectory="neutral",
        significant_changes=["Activity levels decreased over past month"],
    )


def _generate_concerning_profile(rng: random.Random, age: int) -> AppleHealthMockData:
    """Generate data for a concerning health profile - HKRS 40-54."""
    return AppleHealthMockData(
        user_id="",
        
        # 1. Activity (25%) - Low, 0-10/25 points
        daily_steps_avg=rng.randint(3500, 5000),
        active_energy_burned_avg=rng.uniform(150, 250),
        activity_days_with_data=rng.randint(70, 95),
        
        # 2. Fitness (20%) - Below average VO2, 0-8/20 points
        vo2_max=rng.uniform(22, 30),  # <25th percentile
        vo2_max_readings=rng.randint(2, 4),
        
        # 3. Vitals (20%) - Elevated, 5-10/20 points
        resting_hr_avg=rng.randint(78, 85),
        hrv_avg_ms=rng.uniform(22, 32),
        elevated_hr_events=rng.randint(10, 20),
        irregular_rhythm_events=rng.randint(1, 3),
        heart_rate_days_with_data=rng.randint(60, 80),
        
        # 4. Sleep (15%) - Poor, 3-6/15 points
        avg_sleep_duration_hours=rng.uniform(5.2, 6.0),
        sleep_consistency_variance_hours=rng.uniform(1.8, 2.5),
        sleep_nights_with_data=rng.randint(60, 80),
        
        # 5. Body Metrics (10%) - Overweight, 0-5/10 points
        bmi=rng.uniform(29, 33),
        bmi_trend="mild_increase",
        weight_kg=rng.uniform(90, 110),
        height_cm=rng.uniform(165, 185),
        
        # 6. Mobility (10%) - Below average, 2-5/10 points
        walking_speed_avg=rng.uniform(0.95, 1.1),
        walking_steadiness="slightly_unsteady",
        double_support_time_pct=rng.uniform(30, 35),
        
        # 7. Exercise
        workout_frequency_weekly=rng.randint(0, 1),
        workout_avg_duration_minutes=rng.randint(15, 25),
        workout_types=["walking"],
        
        # Trends - negative
        activity_trend_6mo="declining",
        weight_trend_6mo="mild_increase",
        overall_health_trajectory="negative",
        significant_changes=[
            "Resting heart rate increased over past 2 months",
            "Sleep duration decreased significantly",
        ],
    )


def _generate_high_risk_profile(rng: random.Random, age: int) -> AppleHealthMockData:
    """Generate data for a high risk health profile - HKRS <40 (substandard)."""
    return AppleHealthMockData(
        user_id="",
        
        # 1. Activity (25%) - Very low, 0/25 points
        daily_steps_avg=rng.randint(1500, 3500),
        active_energy_burned_avg=rng.uniform(80, 150),
        activity_days_with_data=rng.randint(40, 70),
        
        # 2. Fitness (20%) - Poor VO2, 0/20 points
        vo2_max=rng.uniform(18, 22),  # <25th percentile
        vo2_max_readings=rng.randint(1, 3),
        
        # 3. Vitals (20%) - High HR, low HRV, 0-5/20 points
        resting_hr_avg=rng.randint(85, 98),
        hrv_avg_ms=rng.uniform(12, 22),
        elevated_hr_events=rng.randint(25, 50),
        irregular_rhythm_events=rng.randint(5, 12),
        heart_rate_days_with_data=rng.randint(50, 70),
        
        # 4. Sleep (15%) - Very poor, 0-3/15 points
        avg_sleep_duration_hours=rng.uniform(4.5, 5.2),
        sleep_consistency_variance_hours=rng.uniform(2.5, 4.0),
        sleep_nights_with_data=rng.randint(40, 65),
        
        # 5. Body Metrics (10%) - Obese, 0/10 points
        bmi=rng.uniform(33, 42),
        bmi_trend="significant_increase",
        weight_kg=rng.uniform(105, 140),
        height_cm=rng.uniform(165, 185),
        
        # 6. Mobility (10%) - Poor, 0-2/10 points
        walking_speed_avg=rng.uniform(0.75, 0.95),
        walking_steadiness="unsteady",
        double_support_time_pct=rng.uniform(35, 42),
        
        # 7. Exercise
        workout_frequency_weekly=0,
        workout_avg_duration_minutes=0,
        workout_types=[],
        
        # Trends - very negative
        activity_trend_6mo="declining",
        weight_trend_6mo="significant_increase",
        overall_health_trajectory="negative",
        significant_changes=[
            "Resting heart rate consistently elevated above 85 bpm",
            "Multiple irregular rhythm events detected - medical review recommended",
            "Severe sleep deficiency detected",
            "Very low activity levels - virtually sedentary",
            "BMI indicates obesity - increased cardiovascular risk",
        ],
    )
