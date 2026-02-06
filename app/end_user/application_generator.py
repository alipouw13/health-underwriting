"""
Apple Health Application Form Generator

Generates life insurance application forms focused on Apple Health data.
NO lab results, family history, or substance use - only HealthKit metrics.

The generated application uses the 7 Apple Health categories:
- Activity (25%), Fitness (20%), Vitals (20%), Sleep (15%), 
- Body Metrics (10%), Mobility (10%), Exercise
"""

import json
import logging
import random
from datetime import date, datetime
from typing import Any, Dict, Optional

from app.openai_client import chat_completion
from app.config import load_settings

logger = logging.getLogger("underwriting_assistant")


# Apple Health focused application template
APPLE_HEALTH_APPLICATION_TEMPLATE = """
# Life Insurance Application
## Apple Health Connected Assessment

### Applicant Information
- **Full Name**: {full_name}
- **Date of Birth**: {dob}
- **Age**: {age} years
- **Biological Sex**: {gender}

### Policy Request
- **Policy Type**: {policy_type}
- **Coverage Amount**: ${coverage_amount:,.0f}

---

## Apple Health Data Summary
*Data synced from Apple HealthKit - {data_period} days of measurements*

### 1. Activity (25% HKRS Weight)
- **Daily Steps Average**: {daily_steps:,} steps/day
- **Active Energy Burned**: {active_energy:.0f} kcal/day
- **Activity Trend (6 months)**: {activity_trend}
- **Days with Data**: {activity_days} days

### 2. Fitness (20% HKRS Weight)
- **VO2 Max**: {vo2_max:.1f} mL/kg/min
- **Cardio Fitness Level**: {fitness_level}
- **VO2 Max Readings**: {vo2_readings} measurements

### 3. Vitals (20% HKRS Weight)
- **Resting Heart Rate**: {resting_hr} bpm
- **Heart Rate Variability (HRV)**: {hrv_avg:.0f} ms
- **Irregular Rhythm Events**: {irregular_events}
- **Days with Data**: {vitals_days} days

### 4. Sleep (15% HKRS Weight)
- **Average Sleep Duration**: {sleep_hours:.1f} hours/night
- **Sleep Consistency**: ±{sleep_variance:.1f} hours variance
- **Nights with Data**: {sleep_days} nights

### 5. Body Metrics (10% HKRS Weight)
- **Current BMI**: {bmi:.1f}
- **Weight**: {weight_kg:.1f} kg
- **Height**: {height_cm:.1f} cm
- **BMI Trend**: {bmi_trend}

### 6. Mobility (10% HKRS Weight)
- **Walking Speed**: {walking_speed:.2f} m/s
- **Walking Steadiness**: {walking_steadiness}
- **Double Support Time**: {double_support:.0f}%

### 7. Exercise
- **Weekly Workout Sessions**: {workout_frequency} sessions
- **Average Workout Duration**: {workout_duration} minutes
- **Workout Types**: {workout_types}

---

## Data Quality Assessment
- **Measurement Period**: {data_period} days
- **Last Sync Date**: {last_sync_date}
- **Data Completeness**: {data_completeness}

## Consent & Privacy
✓ Applicant has authorized access to Apple Health data
✓ Data is anonymized and used only for underwriting assessment
✓ No reproductive health, mental health, or location data accessed

---
*Application generated from Apple HealthKit data on {generation_date}*
"""


async def generate_application_document(
    user_profile: Dict[str, Any],
    apple_health_data: Dict[str, Any],
    policy_type: str = "term_life",
    coverage_amount: float = 500000,
) -> str:
    """
    Generate an Apple Health-focused application document.
    
    NO lab results, family history, or substance use - ONLY HealthKit data.
    """
    logger.info(
        "Generating Apple Health application for user %s %s",
        user_profile.get("first_name", "Unknown"),
        user_profile.get("last_name", "")
    )
    
    # Extract activity data
    activity = apple_health_data.get("activity", {})
    daily_steps = activity.get("daily_steps_avg", apple_health_data.get("daily_steps_avg", 8000))
    active_energy = activity.get("active_energy_burned_avg", apple_health_data.get("active_energy_burned_avg", 400))
    activity_trend = activity.get("trend_6mo", apple_health_data.get("activity_trend_weekly", "stable")).replace("_", " ").title()
    activity_days = activity.get("days_with_data", 120)
    
    # Extract fitness data
    fitness = apple_health_data.get("fitness", {})
    vo2_max = fitness.get("vo2_max", apple_health_data.get("vo2_max", 38.0)) or 38.0
    vo2_readings = fitness.get("vo2_max_readings", 5)
    
    # Determine fitness level based on VO2 max
    if vo2_max >= 45:
        fitness_level = "Excellent"
    elif vo2_max >= 38:
        fitness_level = "Good"
    elif vo2_max >= 30:
        fitness_level = "Average"
    else:
        fitness_level = "Below Average"
    
    # Extract vitals data
    heart_rate = apple_health_data.get("heart_rate", {})
    resting_hr = heart_rate.get("resting_hr_avg", apple_health_data.get("resting_hr_avg", 68))
    hrv_avg = heart_rate.get("hrv_avg_ms", apple_health_data.get("hrv_avg_ms", 42)) or 42
    irregular_events = heart_rate.get("irregular_rhythm_events", 0)
    vitals_days = heart_rate.get("days_with_data", 90)
    
    # Extract sleep data
    sleep = apple_health_data.get("sleep", {})
    sleep_hours = sleep.get("avg_sleep_duration_hours", apple_health_data.get("avg_sleep_duration_hours", 7.2)) or 7.2
    sleep_variance = sleep.get("sleep_consistency_variance_hours", 0.8) or 0.8
    sleep_days = sleep.get("nights_with_data", 100)
    
    # Extract body metrics
    body = apple_health_data.get("body_metrics", {})
    bmi = body.get("bmi", apple_health_data.get("bmi", 24.5)) or 24.5
    weight_kg = body.get("weight_kg", apple_health_data.get("weight_kg", 75.0)) or 75.0
    height_cm = body.get("height_cm", apple_health_data.get("height_cm", 175.0)) or 175.0
    bmi_trend = body.get("bmi_trend", "stable").replace("_", " ").title()
    
    # Extract mobility data
    mobility = apple_health_data.get("mobility", {})
    walking_speed = mobility.get("walking_speed_avg", 1.3) or 1.3
    walking_steadiness = mobility.get("walking_steadiness", "normal").replace("_", " ").title()
    double_support = mobility.get("double_support_time_pct", 25) or 25
    
    # Extract exercise data
    exercise = apple_health_data.get("exercise", {})
    workout_frequency = exercise.get("workout_frequency_weekly", apple_health_data.get("weekly_exercise_sessions", 3)) or 3
    workout_duration = exercise.get("workout_avg_duration_minutes", 45) or 45
    workout_types_list = exercise.get("workout_types", ["walking", "running"])
    workout_types = ", ".join(workout_types_list) if isinstance(workout_types_list, list) else str(workout_types_list)
    
    # Build the application document
    full_name = f"{user_profile.get('first_name', 'Unknown')} {user_profile.get('last_name', '')}".strip()
    
    document = APPLE_HEALTH_APPLICATION_TEMPLATE.format(
        full_name=full_name,
        dob=user_profile.get("date_of_birth", "Unknown"),
        age=user_profile.get("age", 35),
        gender=str(user_profile.get("biological_sex", "Unknown")).capitalize(),
        policy_type=policy_type.replace("_", " ").title(),
        coverage_amount=coverage_amount,
        data_period=365,
        daily_steps=daily_steps,
        active_energy=active_energy,
        activity_trend=activity_trend,
        activity_days=activity_days,
        vo2_max=vo2_max,
        fitness_level=fitness_level,
        vo2_readings=vo2_readings,
        resting_hr=resting_hr,
        hrv_avg=hrv_avg,
        irregular_events=irregular_events,
        vitals_days=vitals_days,
        sleep_hours=sleep_hours,
        sleep_variance=sleep_variance,
        sleep_days=sleep_days,
        bmi=bmi,
        weight_kg=weight_kg,
        height_cm=height_cm,
        bmi_trend=bmi_trend,
        walking_speed=walking_speed,
        walking_steadiness=walking_steadiness,
        double_support=double_support,
        workout_frequency=workout_frequency,
        workout_duration=workout_duration,
        workout_types=workout_types,
        data_completeness="High" if activity_days >= 90 else "Medium" if activity_days >= 30 else "Low",
        last_sync_date=datetime.now().strftime("%Y-%m-%d"),
        generation_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    
    logger.info(
        "Generated Apple Health application (%d chars) for %s",
        len(document), full_name
    )
    
    return document


async def generate_and_extract_application(
    user_profile: Dict[str, Any],
    apple_health_data: Dict[str, Any],
    policy_type: str = "term_life",
    coverage_amount: float = 500000,
) -> Dict[str, Any]:
    """
    Generate Apple Health application document and extract structured data.
    
    IMPORTANT: This generates data for APPLE HEALTH workflow ONLY.
    NO lab results, family history, substance use, or medical history.
    ONLY the 7 Apple Health categories are included.
    
    Returns a dict with:
    - document_markdown: The generated application text
    - llm_outputs: Structured extracted data (Apple Health only)
    - extracted_fields: Key fields for display
    """
    # Generate the Apple Health application document
    document_markdown = await generate_application_document(
        user_profile=user_profile,
        apple_health_data=apple_health_data,
        policy_type=policy_type,
        coverage_amount=coverage_amount,
    )
    
    # Build structured llm_outputs - APPLE HEALTH DATA ONLY
    full_name = f"{user_profile.get('first_name', 'Unknown')} {user_profile.get('last_name', '')}".strip()
    age = user_profile.get("age", 35)
    gender = user_profile.get("biological_sex", "unknown")
    
    # Extract health data from nested structure or flat structure
    activity = apple_health_data.get("activity", {})
    fitness = apple_health_data.get("fitness", {})
    heart_rate = apple_health_data.get("heart_rate", {})
    sleep = apple_health_data.get("sleep", {})
    body = apple_health_data.get("body_metrics", {})
    mobility = apple_health_data.get("mobility", {})
    exercise = apple_health_data.get("exercise", {})
    
    # Get values with fallbacks
    bmi = body.get("bmi", apple_health_data.get("bmi", 24.5)) or 24.5
    daily_steps = activity.get("daily_steps_avg", apple_health_data.get("daily_steps_avg", 8000)) or 8000
    resting_hr = heart_rate.get("resting_hr_avg", apple_health_data.get("resting_hr_avg", 68)) or 68
    vo2_max = fitness.get("vo2_max", apple_health_data.get("vo2_max", 38.0)) or 38.0
    sleep_hours = sleep.get("avg_sleep_duration_hours", apple_health_data.get("avg_sleep_duration_hours", 7.2)) or 7.2
    
    # Determine fitness level
    if vo2_max >= 45:
        fitness_level = "Excellent"
    elif vo2_max >= 38:
        fitness_level = "Good"
    elif vo2_max >= 30:
        fitness_level = "Average"
    else:
        fitness_level = "Below Average"
    
    # Build Apple Health-focused patient summary
    summary_text = f"{full_name} is a {age}-year-old {gender} with a BMI of {bmi:.1f}. "
    summary_text += f"Apple Health shows an average of {daily_steps:,} daily steps with a resting heart rate of {resting_hr} bpm. "
    summary_text += f"VO2 Max indicates {fitness_level.lower()} cardio fitness at {vo2_max:.1f} mL/kg/min. "
    summary_text += f"Average sleep duration is {sleep_hours:.1f} hours per night."
    
    # Structure health metrics for the 7 categories
    structured_health_metrics = {
        "patient_id": user_profile.get("user_id", "unknown"),
        "data_source": "apple_health",
        
        # 1. Activity (25% weight)
        "activity": {
            "daily_steps_avg": activity.get("daily_steps_avg", apple_health_data.get("daily_steps_avg", 8000)),
            "active_energy_burned_avg": activity.get("active_energy_burned_avg", 400),
            "days_with_data": activity.get("days_with_data", 120),
            "trend_6mo": activity.get("trend_6mo", "stable"),
        },
        
        # 2. Fitness (20% weight)
        "fitness": {
            "vo2_max": vo2_max,
            "vo2_max_readings": fitness.get("vo2_max_readings", 5),
            "cardio_fitness_level": fitness_level,
        },
        
        # 3. Vitals (20% weight)
        "heart_rate": {
            "resting_hr_avg": resting_hr,
            "hrv_avg_ms": heart_rate.get("hrv_avg_ms", apple_health_data.get("hrv_avg_ms", 42)),
            "elevated_hr_events": heart_rate.get("elevated_hr_events", 0),
            "irregular_rhythm_events": heart_rate.get("irregular_rhythm_events", 0),
            "days_with_data": heart_rate.get("days_with_data", 90),
        },
        
        # 4. Sleep (15% weight)
        "sleep": {
            "avg_sleep_duration_hours": sleep_hours,
            "sleep_consistency_variance_hours": sleep.get("sleep_consistency_variance_hours", 0.8),
            "nights_with_data": sleep.get("nights_with_data", 100),
        },
        
        # 5. Body Metrics (10% weight)
        "body_metrics": {
            "bmi": bmi,
            "bmi_trend": body.get("bmi_trend", "stable"),
            "weight_kg": body.get("weight_kg", apple_health_data.get("weight_kg", 75)),
            "height_cm": body.get("height_cm", apple_health_data.get("height_cm", 175)),
        },
        
        # 6. Mobility (10% weight)
        "mobility": {
            "walking_speed_avg": mobility.get("walking_speed_avg", 1.3),
            "walking_steadiness": mobility.get("walking_steadiness", "normal"),
            "double_support_time_pct": mobility.get("double_support_time_pct", 25),
        },
        
        # 7. Exercise
        "exercise": {
            "workout_frequency_weekly": exercise.get("workout_frequency_weekly", 3),
            "workout_avg_duration_minutes": exercise.get("workout_avg_duration_minutes", 45),
            "workout_types": exercise.get("workout_types", ["walking", "running"]),
        },
    }
    
    llm_outputs = {
        "application_summary": {
            "patient_id": user_profile.get("user_id", "unknown"),
            "customer_profile": {
                "parsed": {
                    "full_name": full_name,
                    "date_of_birth": str(user_profile.get("date_of_birth", "")),
                    "age": age,
                    "gender": gender,
                    "summary": summary_text,
                    "key_fields": [
                        {"label": "Full Name", "value": full_name},
                        {"label": "Age", "value": str(age)},
                        {"label": "Gender", "value": str(gender).capitalize()},
                    ],
                },
                "summary": summary_text,
            },
        },
        "patient_summary": {
            "patient_id": user_profile.get("user_id", "unknown"),
            "name": full_name,
            "age": age,
            "gender": gender,
            "policy_type": policy_type,
            "coverage_amount": coverage_amount,
            "summary": summary_text,
        },
        "patient_profile": {
            "name": full_name,
            "date_of_birth": str(user_profile.get("date_of_birth", "")),
            "age": age,
            "gender": gender,
            "height_cm": body.get("height_cm", apple_health_data.get("height_cm", 175)),
            "weight_kg": body.get("weight_kg", apple_health_data.get("weight_kg", 75)),
            "bmi": bmi,
            "policy_type_requested": policy_type,
            "coverage_amount_requested": coverage_amount,
        },
        # The 7 Apple Health categories - this is the ONLY health data
        "health_metrics": structured_health_metrics,
        
        # NO lab_results, family_history, substance_use, medical_summary, etc.
        # These are explicitly NOT included for Apple Health workflow
        
        # Workflow routing flags
        "source": "end_user",
        "persona": "end_user",
        "workflow_type": "apple_health",
        "ingestion_type": "apple_health_application",
        "is_apple_health": True,  # Flag for UI to show Apple Health layout
    }
    
    # Build extracted_fields for display - Apple Health focused
    extracted_fields = {
        "applicant_name": full_name,
        "ApplicantName": full_name,
        "applicant_age": age,
        "Age": str(age),
        "applicant_dob": str(user_profile.get("date_of_birth", "")),
        "DateOfBirth": str(user_profile.get("date_of_birth", "")),
        "biological_sex": gender,
        "Gender": str(gender).capitalize() if gender else "Unknown",
        "height": f"{body.get('height_cm', 175):.1f} cm",
        "Height": f"{body.get('height_cm', 175):.1f} cm",
        "weight": f"{body.get('weight_kg', 75):.1f} kg",
        "Weight": f"{body.get('weight_kg', 75):.1f} kg",
        "bmi": round(bmi, 2),
        "BMI": round(bmi, 2),
        "policy_type": policy_type,
        "coverage_amount": coverage_amount,
        "data_source": "apple_health",
        
        # Apple Health specific fields
        "daily_steps": daily_steps,
        "resting_hr": resting_hr,
        "vo2_max": vo2_max,
        "sleep_hours": sleep_hours,
        "fitness_level": fitness_level,
    }
    
    return {
        "document_markdown": document_markdown,
        "llm_outputs": llm_outputs,
        "extracted_fields": extracted_fields,
    }
