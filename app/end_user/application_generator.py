"""
LLM-based Application Form Generator

Generates realistic life insurance application forms using LLM
based on user profile and Apple Health data.

The generated application document is then processed through
the same Content Understanding pipeline as admin uploads.
"""

import json
import logging
import random
from datetime import date, datetime
from typing import Any, Dict, Optional

from app.openai_client import chat_completion
from app.config import load_settings

logger = logging.getLogger("underwriting_assistant")


SAMPLE_APPLICATION_TEMPLATE = """
# Life Insurance Application Form

## Applicant Information
- **Full Name**: John Smith
- **Date of Birth**: 1985-03-15
- **Age**: 40
- **Gender**: Male
- **Address**: 123 Main Street, Anytown, CA 90210
- **Phone**: (555) 123-4567
- **Email**: john.smith@email.com
- **Occupation**: Software Engineer
- **Employer**: Tech Company Inc.
- **Annual Income**: $125,000

## Policy Details
- **Policy Type Requested**: Term Life Insurance
- **Coverage Amount Requested**: $500,000
- **Term Length**: 20 years
- **Beneficiary**: Jane Smith (Spouse)

## Health History

### Current Health Status
- **Height**: 178 cm
- **Weight**: 82 kg
- **BMI**: 25.9
- **Blood Pressure**: 122/78 mmHg
- **Resting Heart Rate**: 72 bpm

### Medical History
- No history of heart disease
- No history of cancer
- No history of diabetes
- No major surgeries in past 5 years

### Lifestyle Factors
- **Smoking Status**: Non-smoker
- **Alcohol Consumption**: Social drinker (2-3 drinks per week)
- **Exercise Frequency**: 3-4 times per week
- **Average Daily Steps**: 8,500

### Family Medical History
- Father: Hypertension, diagnosed at age 55
- Mother: No significant conditions

### Current Medications
- None

### Recent Lab Results
- Total Cholesterol: 195 mg/dL
- Fasting Glucose: 92 mg/dL

## Connected Health Data (Apple Health)
The applicant has authorized access to their Apple Health data which shows:
- **30-Day Average Daily Steps**: 8,500
- **Average Resting Heart Rate**: 72 bpm
- **Average Sleep Duration**: 7.2 hours
- **Heart Rate Variability (HRV)**: 45 ms
- **Activity Trend**: Stable
- **Sleep Quality**: Good

## Declarations
I hereby declare that all information provided is true and complete to the best of my knowledge.

**Signature**: John Smith
**Date**: 2024-01-15
"""

GENERATION_PROMPT = """You are a life insurance application form generator. Generate a realistic, detailed life insurance application form based on the provided user data.

The application should follow this exact structure (use the template format):

1. **Applicant Information** - Name, DOB, age, gender, contact, occupation, income
2. **Policy Details** - Type, coverage amount, term, beneficiary  
3. **Health History**:
   - Current Health Status (height, weight, BMI, vitals)
   - Medical History (conditions, surgeries)
   - Lifestyle Factors (smoking, alcohol, exercise)
   - Family Medical History
   - Current Medications
   - Recent Lab Results
4. **Connected Health Data (Apple Health)** - Real-time health metrics
5. **Declarations** - Signature and date

## User Data to Use:

### Personal Information:
{personal_info}

### Apple Health Metrics:
{health_metrics}

### Policy Request:
{policy_request}

## Important Instructions:
1. Generate REALISTIC but SYNTHETIC data for any fields not provided
2. Make the health history consistent with the Apple Health metrics
3. Include some minor health findings to make it realistic (slightly elevated BMI, borderline cholesterol, family history of common conditions)
4. Do NOT make everything perfect - real applications have some risk factors
5. Use the exact markdown format with headers and bullet points
6. Include realistic lab values that align with the health metrics
7. Add 1-2 minor medical conditions or lifestyle factors based on the age and health metrics
8. Generate a realistic occupation and income based on the age

Generate ONLY the application form document in markdown format. No explanations or additional text."""


async def generate_application_document(
    user_profile: Dict[str, Any],
    apple_health_data: Dict[str, Any],
    policy_type: str = "term_life",
    coverage_amount: float = 500000,
) -> str:
    """
    Generate a realistic life insurance application document using LLM.
    
    Args:
        user_profile: User's personal information (name, DOB, gender, etc.)
        apple_health_data: Mock Apple Health metrics
        policy_type: Type of insurance policy requested
        coverage_amount: Coverage amount requested
        
    Returns:
        Markdown-formatted application document
    """
    logger.info(
        "Generating application document for user %s %s",
        user_profile.get("first_name", "Unknown"),
        user_profile.get("last_name", "")
    )
    
    # Format personal info
    personal_info = f"""- Full Name: {user_profile.get('first_name', 'Unknown')} {user_profile.get('last_name', '')}
- Date of Birth: {user_profile.get('date_of_birth', 'Unknown')}
- Age: {user_profile.get('age', 35)}
- Biological Sex: {user_profile.get('biological_sex', 'Unknown')}"""

    # Format health metrics
    health_metrics = f"""- Average Daily Steps: {apple_health_data.get('daily_steps_avg', 8000)}
- Resting Heart Rate: {apple_health_data.get('resting_hr_avg', 68)} bpm
- Average Sleep Duration: {apple_health_data.get('avg_sleep_duration_hours', 7.2)} hours
- BMI: {apple_health_data.get('bmi', 24.5)}
- Height: {apple_health_data.get('height_cm', 170)} cm
- Weight: {apple_health_data.get('weight_kg', 70)} kg
- Heart Rate Variability: {apple_health_data.get('hrv_avg_ms', 42)} ms
- Weekly Exercise Sessions: {apple_health_data.get('weekly_exercise_sessions', 3)}
- Activity Trend: {apple_health_data.get('activity_trend_weekly', 'stable')}
- Elevated Heart Rate Events (30 days): {apple_health_data.get('elevated_hr_events', 0)}
- Sleep Efficiency: {apple_health_data.get('sleep_efficiency_pct', 88)}%"""

    # Format policy request
    policy_request = f"""- Policy Type: {policy_type.replace('_', ' ').title()}
- Coverage Amount: ${coverage_amount:,.0f}"""

    # Build the prompt
    prompt = GENERATION_PROMPT.format(
        personal_info=personal_info,
        health_metrics=health_metrics,
        policy_request=policy_request,
    )
    
    try:
        # Call LLM to generate the application
        settings = load_settings()
        response = chat_completion(
            settings=settings.openai,
            messages=[
                {"role": "system", "content": "You are a life insurance application form generator. Generate realistic, detailed applications based on user data."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,  # Some creativity for realistic variation
            max_tokens=2000,
        )
        
        # Extract content from response
        generated_doc = response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        # Ensure it starts with a proper header
        if not generated_doc.startswith("# "):
            generated_doc = "# Life Insurance Application Form\n\n" + generated_doc
        
        logger.info(
            "Generated application document (%d characters) for %s %s",
            len(generated_doc),
            user_profile.get("first_name", "Unknown"),
            user_profile.get("last_name", "")
        )
        
        return generated_doc
        
    except Exception as e:
        logger.error("Failed to generate application document: %s", e, exc_info=True)
        # Fall back to template-based generation
        return _generate_fallback_document(user_profile, apple_health_data, policy_type, coverage_amount)


def _generate_fallback_document(
    user_profile: Dict[str, Any],
    apple_health_data: Dict[str, Any],
    policy_type: str,
    coverage_amount: float,
) -> str:
    """Generate a template-based fallback document if LLM fails."""
    
    full_name = f"{user_profile.get('first_name', 'Unknown')} {user_profile.get('last_name', '')}"
    dob = user_profile.get('date_of_birth', 'Unknown')
    age = user_profile.get('age', 35)
    gender = user_profile.get('biological_sex', 'Unknown')
    
    bmi = apple_health_data.get('bmi', 24.5)
    height = apple_health_data.get('height_cm', 170)
    weight = apple_health_data.get('weight_kg', 70)
    resting_hr = apple_health_data.get('resting_hr_avg', 68)
    daily_steps = apple_health_data.get('daily_steps_avg', 8000)
    sleep_hours = apple_health_data.get('avg_sleep_duration_hours', 7.2)
    
    return f"""# Life Insurance Application Form

## Applicant Information
- **Full Name**: {full_name}
- **Date of Birth**: {dob}
- **Age**: {age}
- **Gender**: {gender.capitalize() if isinstance(gender, str) else gender}
- **Occupation**: Professional
- **Annual Income**: $75,000

## Policy Details
- **Policy Type Requested**: {policy_type.replace('_', ' ').title()}
- **Coverage Amount Requested**: ${coverage_amount:,.0f}
- **Term Length**: 20 years

## Health History

### Current Health Status
- **Height**: {height} cm
- **Weight**: {weight} kg
- **BMI**: {bmi:.1f}
- **Resting Heart Rate**: {resting_hr} bpm

### Medical History
- No significant medical history reported
- No major surgeries

### Lifestyle Factors
- **Smoking Status**: Non-smoker
- **Alcohol Consumption**: Occasional
- **Exercise Frequency**: Regular ({apple_health_data.get('weekly_exercise_sessions', 3)} sessions/week)
- **Average Daily Steps**: {daily_steps:,}

### Family Medical History
- No significant family history reported

### Current Medications
- None reported

## Connected Health Data (Apple Health)
- **30-Day Average Daily Steps**: {daily_steps:,}
- **Average Resting Heart Rate**: {resting_hr} bpm
- **Average Sleep Duration**: {sleep_hours:.1f} hours
- **Heart Rate Variability (HRV)**: {apple_health_data.get('hrv_avg_ms', 42)} ms
- **Activity Trend**: {apple_health_data.get('activity_trend_weekly', 'stable').capitalize()}
- **Sleep Efficiency**: {apple_health_data.get('sleep_efficiency_pct', 88)}%

## Declarations
I hereby declare that all information provided is true and complete to the best of my knowledge.

**Date**: {datetime.now().strftime('%Y-%m-%d')}
"""


async def generate_and_extract_application(
    user_profile: Dict[str, Any],
    apple_health_data: Dict[str, Any],
    policy_type: str = "term_life",
    coverage_amount: float = 500000,
) -> Dict[str, Any]:
    """
    Generate application document and extract structured data.
    
    This simulates what Content Understanding would do:
    1. Generate the application document (like receiving a PDF)
    2. Extract structured data from it (like CU extraction)
    
    Returns a dict with:
    - document_markdown: The generated application text
    - llm_outputs: Structured extracted data
    - extracted_fields: Key fields for display
    """
    import random
    
    # Generate the application document
    document_markdown = await generate_application_document(
        user_profile=user_profile,
        apple_health_data=apple_health_data,
        policy_type=policy_type,
        coverage_amount=coverage_amount,
    )
    
    # Build structured llm_outputs similar to what extraction produces
    full_name = f"{user_profile.get('first_name', 'Unknown')} {user_profile.get('last_name', '')}".strip()
    age = user_profile.get("age", 35)
    gender = user_profile.get("biological_sex", "unknown")
    bmi = apple_health_data.get("bmi", 24.5)
    
    # Generate realistic occupation based on age
    occupations = [
        "Software Engineer", "Marketing Manager", "Healthcare Professional",
        "Financial Analyst", "Teacher", "Sales Executive", "Consultant",
        "Business Owner", "Project Manager", "Engineer"
    ]
    occupation = random.choice(occupations)
    
    # Generate realistic lab values based on BMI and age
    base_cholesterol = 180 + (bmi - 22) * 3 + (age - 30) * 0.5
    cholesterol_total = round(base_cholesterol + random.uniform(-15, 15), 1)
    glucose_fasting = round(88 + (bmi - 22) * 1.5 + random.uniform(-5, 10), 1)
    
    # Determine smoking status based on health metrics
    smoking_status = "non-smoker"
    if apple_health_data.get("resting_hr_avg", 68) > 80:
        smoking_status = random.choice(["non-smoker", "former smoker"])
    
    # Build patient summary text
    summary_text = f"{full_name} is a {age}-year-old {gender} {occupation} with a {'low' if bmi < 25 else 'moderate' if bmi < 30 else 'elevated'} BMI of {bmi:.1f}. "
    summary_text += f"They are a {smoking_status} who exercises regularly with an average of {apple_health_data.get('daily_steps_avg', 8000):,} daily steps. "
    summary_text += f"Lab results show total cholesterol of {cholesterol_total} mg/dL and fasting glucose of {glucose_fasting} mg/dL."
    
    llm_outputs = {
        "application_summary": {
            "patient_id": user_profile.get("user_id", "unknown"),
            "customer_profile": {
                "parsed": {
                    "full_name": full_name,
                    "date_of_birth": str(user_profile.get("date_of_birth", "")),
                    "age": age,
                    "gender": gender,
                    "occupation": occupation,
                    "smoking_status": smoking_status,
                    "alcohol_use": "occasional",
                    "summary": summary_text,  # Summary inside parsed for PatientSummary component
                    "key_fields": [
                        {"label": "Full Name", "value": full_name},
                        {"label": "Age", "value": str(age)},
                        {"label": "Gender", "value": gender.capitalize()},
                        {"label": "Occupation", "value": occupation},
                    ],
                },
                "summary": summary_text,  # Also keep at this level for backward compatibility
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
            "height_cm": round(apple_health_data.get("height_cm", 170), 2),
            "weight_kg": round(apple_health_data.get("weight_kg", 70), 2),
            "bmi": round(bmi, 2),
            "occupation": occupation,
            "policy_type_requested": policy_type,
            "coverage_amount_requested": coverage_amount,
        },
        "health_metrics": apple_health_data,
        # Medical summary with lab results - structured for panel extraction
        "medical_summary": {
            # Cholesterol data - structured for LabResultsPanel
            "high_cholesterol": {
                "parsed": {
                    "lipid_panels": [
                        {
                            "total_cholesterol": cholesterol_total,
                            "ldl": round(cholesterol_total * 0.6 + random.uniform(-10, 10), 1),  # ~60% of total
                            "hdl": round(50 + random.uniform(-5, 15), 1),  # 45-65 typical
                            "triglycerides": round(120 + (bmi - 22) * 5 + random.uniform(-20, 20), 1),
                            "date": datetime.now().strftime("%Y-%m-%d"),
                        }
                    ],
                    "summary": f"Total cholesterol: {cholesterol_total} mg/dL",
                    "risk_assessment": "normal" if cholesterol_total < 200 else "borderline" if cholesterol_total < 240 else "elevated",
                }
            },
            # Blood pressure - structured for LabResultsPanel
            "hypertension": {
                "parsed": {
                    "bp_readings": _generate_bp_readings(bmi, age),
                    "summary": "Blood pressure within normal range" if bmi < 27 else "Borderline elevated blood pressure",
                    "risk_assessment": "normal" if bmi < 27 else "monitor",
                }
            },
            # Diabetes/glucose - structured for LabResultsPanel
            "diabetes": {
                "parsed": {
                    "glucose_readings": [
                        {
                            "value": glucose_fasting,
                            "date": datetime.now().strftime("%Y-%m-%d"),
                            "test_type": "fasting"
                        }
                    ],
                    "a1c_readings": [
                        {
                            "value": round(5.2 + (glucose_fasting - 90) * 0.02, 1),
                            "date": datetime.now().strftime("%Y-%m-%d"),
                        }
                    ],
                    "summary": f"Fasting glucose: {glucose_fasting} mg/dL",
                }
            },
            # Family history - structured for FamilyHistoryPanel
            "family_history": {
                "parsed": {
                    "relatives": _generate_family_history_relatives(age),
                    "summary": _generate_family_history_summary(age),
                    "risk_assessment": "low" if age < 40 else "moderate",
                }
            },
            # Other medical findings - structured for SubstanceUsePanel
            "other_medical_findings": {
                "parsed": {
                    "lifestyle": {
                        "smoking_status": _get_smoking_status_text(smoking_status),
                        "alcohol": _get_alcohol_status_text(),
                        "marijuana": "No marijuana use reported",
                        "other": "No other substance use reported",
                    },
                    "allergies": _generate_allergies(),
                    "medications": [],
                }
            },
        },
        # Medical timeline with health data connection event
        "medical_timeline": [
            {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "event": "Apple Health data connected",
                "category": "health_data",
                "description": f"Health metrics synced: BMI {bmi:.1f}, Avg steps {apple_health_data.get('daily_steps_avg', 8000):,}/day"
            },
            {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "event": "Lab results recorded",
                "category": "lab_results",
                "description": f"Cholesterol: {cholesterol_total} mg/dL, Glucose: {glucose_fasting} mg/dL"
            },
        ],
        "diagnoses_conditions": [],
        "medications": [],
        "lab_results": [
            {
                "name": "Total Cholesterol",
                "value": cholesterol_total,
                "unit": "mg/dL",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "status": "normal" if cholesterol_total < 200 else "borderline" if cholesterol_total < 240 else "high"
            },
            {
                "name": "Fasting Glucose",
                "value": glucose_fasting,
                "unit": "mg/dL",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "status": "normal" if glucose_fasting < 100 else "prediabetic" if glucose_fasting < 126 else "diabetic"
            },
        ],
        # Keep these for backward compatibility
        "family_history": _generate_family_history(age),
        "substance_use": {
            "tobacco": {"status": "never" if smoking_status == "non-smoker" else "former", "details": smoking_status},
            "alcohol": {"status": "occasional", "details": "Social drinker, 2-3 drinks per week"},
            "marijuana": {"status": "never", "details": None},
            "drugs": {"status": "never", "details": None},
        },
        "source": "end_user",
        "ingestion_type": "llm_generated_application",
    }
    
    # Build extracted_fields for display (with proper formatting)
    extracted_fields = {
        "applicant_name": full_name,
        "ApplicantName": full_name,
        "applicant_age": age,
        "Age": str(age),
        "applicant_dob": str(user_profile.get("date_of_birth", "")),
        "DateOfBirth": str(user_profile.get("date_of_birth", "")),
        "biological_sex": gender,
        "Gender": gender.capitalize() if gender else "Unknown",
        "height": f"{round(apple_health_data.get('height_cm', 170), 2)} cm",
        "Height": f"{round(apple_health_data.get('height_cm', 170), 2)} cm",
        "weight": f"{round(apple_health_data.get('weight_kg', 70), 2)} kg",
        "Weight": f"{round(apple_health_data.get('weight_kg', 70), 2)} kg",
        "bmi": round(bmi, 2),
        "BMI": round(bmi, 2),
        "occupation": occupation,
        "Occupation": occupation,
        "policy_type": policy_type,
        "coverage_amount": coverage_amount,
        "data_source": "apple_health_llm_generated",
        # Substance use fields for SubstanceUsePanel
        "SmokingStatus": {
            "field_name": "SmokingStatus",
            "value": _get_smoking_status_text(smoking_status),
            "confidence": 0.95,
        },
        "AlcoholUse": {
            "field_name": "AlcoholUse", 
            "value": _get_alcohol_status_text(),
            "confidence": 0.95,
        },
        "DrugUse": {
            "field_name": "DrugUse",
            "value": "No illicit drug use reported",
            "confidence": 0.95,
        },
        # Lab results for LabResultsPanel
        "LipidPanelResults": {
            "field_name": "LipidPanelResults",
            "value": f"Total Cholesterol: {cholesterol_total} mg/dL",
            "confidence": 0.95,
        },
        "BloodPressureReadings": {
            "field_name": "BloodPressureReadings",
            "value": _generate_bp_readings(bmi, age),
            "confidence": 0.95,
        },
        # Family history for FamilyHistoryPanel
        "FamilyHistorySummary": {
            "field_name": "FamilyHistorySummary",
            "value": _generate_family_history_summary(age),
            "confidence": 0.95,
        },
    }
    
    return {
        "document_markdown": document_markdown,
        "llm_outputs": llm_outputs,
        "extracted_fields": extracted_fields,
    }


def _generate_family_history(age: int) -> list:
    """Generate realistic family history based on age."""
    import random
    
    history = []
    
    # Common family history items with age-based probabilities
    possible_conditions = [
        {"condition": "Type 2 Diabetes", "relation": "Mother", "age_onset": 55},
        {"condition": "Hypertension", "relation": "Father", "age_onset": 50},
        {"condition": "Heart Disease", "relation": "Grandfather", "age_onset": 65},
        {"condition": "High Cholesterol", "relation": "Father", "age_onset": 45},
        {"condition": "Type 2 Diabetes", "relation": "Grandmother", "age_onset": 60},
    ]
    
    # Add 1-2 family history items for older applicants
    num_items = 1 if age < 40 else 2 if age < 50 else random.randint(1, 3)
    selected = random.sample(possible_conditions, min(num_items, len(possible_conditions)))
    
    for item in selected:
        history.append({
            "condition": item["condition"],
            "relation": item["relation"],
            "age_at_onset": item["age_onset"],
        })
    
    return history


def _generate_family_history_relatives(age: int) -> list:
    """Generate family history relatives in the format expected by FamilyHistoryPanel."""
    import random
    
    possible_relatives = [
        {"relationship": "Father", "condition": "Hypertension", "age_at_onset": "52", "notes": "Controlled with medication"},
        {"relationship": "Mother", "condition": "Type 2 Diabetes", "age_at_onset": "58", "notes": "Diet-controlled"},
        {"relationship": "Paternal Grandfather", "condition": "Heart Disease", "age_at_death": "72", "notes": "MI at age 68"},
        {"relationship": "Maternal Grandmother", "condition": "Breast Cancer", "age_at_onset": "65", "notes": "Survivor"},
        {"relationship": "Father", "condition": "High Cholesterol", "age_at_onset": "45", "notes": "On statin therapy"},
        {"relationship": "Mother", "condition": "Osteoporosis", "age_at_onset": "62", "notes": ""},
        {"relationship": "Brother", "condition": "None reported", "notes": "Healthy, age 42"},
        {"relationship": "Sister", "condition": "None reported", "notes": "Healthy, age 38"},
    ]
    
    # Select 2-4 relatives based on age
    num_items = 2 if age < 35 else 3 if age < 50 else random.randint(3, 4)
    selected = random.sample(possible_relatives, min(num_items, len(possible_relatives)))
    
    return selected


def _generate_family_history_summary(age: int) -> str:
    """Generate a summary of family history."""
    if age < 35:
        return "Limited family history of common conditions. Father has hypertension."
    elif age < 50:
        return "Family history notable for cardiovascular disease and diabetes. Both parents have chronic conditions managed with medication."
    else:
        return "Significant family history including cardiovascular disease, diabetes, and cancer. Multiple first-degree relatives affected."


def _generate_bp_readings(bmi: float, age: int) -> list:
    """Generate realistic blood pressure readings."""
    import random
    
    # Base BP increases with BMI and age
    base_systolic = 110 + (bmi - 22) * 2 + (age - 30) * 0.3
    base_diastolic = 70 + (bmi - 22) * 1 + (age - 30) * 0.2
    
    readings = []
    for i in range(2):  # Generate 2 readings
        systolic = round(base_systolic + random.uniform(-8, 8))
        diastolic = round(base_diastolic + random.uniform(-5, 5))
        readings.append({
            "systolic": systolic,
            "diastolic": diastolic,
            "date": datetime.now().strftime("%Y-%m-%d"),
        })
    
    return readings


def _get_smoking_status_text(smoking_status: str) -> str:
    """Get descriptive text for smoking status."""
    if smoking_status == "non-smoker":
        return "Non-smoker. Never used tobacco products."
    elif smoking_status == "former smoker":
        return "Former smoker. Quit 5+ years ago. No current tobacco use."
    else:
        return f"{smoking_status.capitalize()}"


def _get_alcohol_status_text() -> str:
    """Get descriptive text for alcohol use."""
    import random
    options = [
        "Social drinker. 2-3 alcoholic beverages per week.",
        "Occasional drinker. 1-2 drinks per week on average.",
        "Light social drinker. Wine with dinner occasionally.",
        "Moderate alcohol consumption. 3-5 drinks per week.",
    ]
    return random.choice(options)


def _generate_allergies() -> list:
    """Generate realistic allergies list."""
    import random
    
    possible_allergies = [
        {"allergen": "Penicillin", "reaction": "Rash", "severity": "moderate"},
        {"allergen": "Shellfish", "reaction": "Anaphylaxis", "severity": "severe"},
        {"allergen": "Pollen", "reaction": "Seasonal allergies", "severity": "mild"},
        {"allergen": "Dust mites", "reaction": "Respiratory", "severity": "mild"},
        {"allergen": "Latex", "reaction": "Contact dermatitis", "severity": "moderate"},
    ]
    
    # 60% chance of having no allergies, 40% chance of 1-2 allergies
    if random.random() < 0.6:
        return []
    
    num_allergies = random.randint(1, 2)
    return random.sample(possible_allergies, num_allergies)
