"""
LLM-based Application Form Generator

Generates realistic life insurance application forms using LLM
based on user profile and Apple Health data.

The generated application document is then processed through
the same Content Understanding pipeline as admin uploads.
"""

import json
import logging
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
                    "key_fields": [
                        {"label": "Full Name", "value": full_name},
                        {"label": "Age", "value": str(age)},
                        {"label": "Gender", "value": gender.capitalize()},
                        {"label": "Occupation", "value": occupation},
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
            "height_cm": round(apple_health_data.get("height_cm", 170), 2),
            "weight_kg": round(apple_health_data.get("weight_kg", 70), 2),
            "bmi": round(bmi, 2),
            "occupation": occupation,
            "policy_type_requested": policy_type,
            "coverage_amount_requested": coverage_amount,
        },
        "health_metrics": apple_health_data,
        # Medical summary with lab results
        "medical_summary": {
            "cholesterol": {
                "parsed": {
                    "lipid_panels": [
                        {
                            "total_cholesterol": cholesterol_total,
                            "date": datetime.now().strftime("%Y-%m-%d"),
                        }
                    ]
                }
            },
            "diabetes": {
                "parsed": {
                    "glucose_readings": [
                        {
                            "value": glucose_fasting,
                            "date": datetime.now().strftime("%Y-%m-%d"),
                            "test_type": "fasting"
                        }
                    ]
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
            }
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
        # Family history (generate some common items)
        "family_history": _generate_family_history(age),
        # Substance use
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
