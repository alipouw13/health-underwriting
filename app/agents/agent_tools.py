"""
Azure AI Foundry Agent Tools

Defines real function tools that agents can call during execution.
These tools follow the Azure AI Foundry Function Calling pattern:
1. Define Python functions with typed parameters
2. Create FunctionToolDefinition schemas for each function
3. Register tools with agents
4. Handle requires_action status and execute tools
5. Submit tool outputs back to the agent

Reference: https://learn.microsoft.com/en-us/azure/ai-foundry/agents/how-to/tools/function-calling
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, date

logger = logging.getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        return super().default(obj)


# =============================================================================
# TOOL FUNCTION DEFINITIONS
# These are the actual functions that get executed when an agent calls a tool
# =============================================================================

def analyze_health_metrics(
    age: int,
    height_cm: float,
    weight_kg: float,
    blood_pressure_systolic: Optional[int] = None,
    blood_pressure_diastolic: Optional[int] = None,
    cholesterol_total: Optional[float] = None,
    glucose_fasting: Optional[float] = None,
) -> str:
    """
    Analyzes health metrics and returns a structured assessment.
    
    :param age: Applicant's age in years
    :param height_cm: Height in centimeters
    :param weight_kg: Weight in kilograms
    :param blood_pressure_systolic: Systolic blood pressure (optional)
    :param blood_pressure_diastolic: Diastolic blood pressure (optional)
    :param cholesterol_total: Total cholesterol mg/dL (optional)
    :param glucose_fasting: Fasting glucose mg/dL (optional)
    :return: JSON string with health metrics analysis
    """
    logger.info("TOOL EXECUTION: analyze_health_metrics(age=%d, height=%s, weight=%s)", 
                age, height_cm, weight_kg)
    
    # Handle missing height/weight with defaults (average adult values)
    if height_cm is None or height_cm <= 0:
        height_cm = 170.0  # Default to average adult height
        logger.warning("analyze_health_metrics: height_cm was None/invalid, using default 170cm")
    if weight_kg is None or weight_kg <= 0:
        weight_kg = 70.0  # Default to average adult weight
        logger.warning("analyze_health_metrics: weight_kg was None/invalid, using default 70kg")
    if age is None or age <= 0:
        age = 35  # Default to average adult age
        logger.warning("analyze_health_metrics: age was None/invalid, using default 35")
    
    # Calculate BMI
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    
    # BMI classification
    if bmi < 18.5:
        bmi_category = "underweight"
        bmi_risk = "moderate"
    elif bmi < 25:
        bmi_category = "normal"
        bmi_risk = "low"
    elif bmi < 30:
        bmi_category = "overweight"
        bmi_risk = "moderate"
    elif bmi < 35:
        bmi_category = "obese_class_1"
        bmi_risk = "elevated"
    elif bmi < 40:
        bmi_category = "obese_class_2"
        bmi_risk = "high"
    else:
        bmi_category = "obese_class_3"
        bmi_risk = "very_high"
    
    # Blood pressure assessment
    bp_assessment = None
    if blood_pressure_systolic and blood_pressure_diastolic:
        if blood_pressure_systolic < 120 and blood_pressure_diastolic < 80:
            bp_assessment = {"category": "normal", "risk": "low"}
        elif blood_pressure_systolic < 130 and blood_pressure_diastolic < 80:
            bp_assessment = {"category": "elevated", "risk": "low"}
        elif blood_pressure_systolic < 140 or blood_pressure_diastolic < 90:
            bp_assessment = {"category": "hypertension_stage_1", "risk": "moderate"}
        elif blood_pressure_systolic < 180 or blood_pressure_diastolic < 120:
            bp_assessment = {"category": "hypertension_stage_2", "risk": "high"}
        else:
            bp_assessment = {"category": "hypertensive_crisis", "risk": "critical"}
    
    # Cholesterol assessment
    cholesterol_assessment = None
    if cholesterol_total:
        if cholesterol_total < 200:
            cholesterol_assessment = {"category": "desirable", "risk": "low"}
        elif cholesterol_total < 240:
            cholesterol_assessment = {"category": "borderline_high", "risk": "moderate"}
        else:
            cholesterol_assessment = {"category": "high", "risk": "elevated"}
    
    # Glucose assessment
    glucose_assessment = None
    if glucose_fasting:
        if glucose_fasting < 100:
            glucose_assessment = {"category": "normal", "risk": "low"}
        elif glucose_fasting < 126:
            glucose_assessment = {"category": "prediabetes", "risk": "moderate"}
        else:
            glucose_assessment = {"category": "diabetes_range", "risk": "high"}
    
    result = {
        "bmi": {
            "value": round(bmi, 1),
            "category": bmi_category,
            "risk_level": bmi_risk
        },
        "age_factor": {
            "value": age,
            "risk_level": "low" if age < 40 else "moderate" if age < 55 else "elevated" if age < 65 else "high"
        },
        "blood_pressure": bp_assessment,
        "cholesterol": cholesterol_assessment,
        "glucose": glucose_assessment,
        "overall_health_risk": _calculate_overall_risk([
            bmi_risk,
            bp_assessment["risk"] if bp_assessment else None,
            cholesterol_assessment["risk"] if cholesterol_assessment else None,
            glucose_assessment["risk"] if glucose_assessment else None,
        ])
    }
    
    return json.dumps(result, indent=2)


def extract_risk_indicators(
    medical_conditions: List[str],
    medications: List[str],
    family_history: List[str],
    lifestyle_factors: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Extracts and categorizes risk indicators from medical history.
    
    :param medical_conditions: List of diagnosed medical conditions
    :param medications: List of current medications
    :param family_history: List of family medical history items
    :param lifestyle_factors: Dict with smoking, alcohol, exercise info
    :return: JSON string with categorized risk indicators
    """
    logger.info("TOOL EXECUTION: extract_risk_indicators(conditions=%d, meds=%d)", 
                len(medical_conditions), len(medications))
    
    # Condition risk mapping
    HIGH_RISK_CONDITIONS = {
        "cancer", "heart_disease", "stroke", "diabetes_type_1", "kidney_disease",
        "liver_disease", "copd", "hiv", "hepatitis_c", "multiple_sclerosis"
    }
    MODERATE_RISK_CONDITIONS = {
        "diabetes_type_2", "hypertension", "asthma", "depression", "anxiety",
        "sleep_apnea", "arthritis", "thyroid_disorder", "gerd"
    }
    
    # Categorize conditions
    high_risk = []
    moderate_risk = []
    low_risk = []
    
    for condition in medical_conditions:
        condition_lower = condition.lower().replace(" ", "_")
        if any(hr in condition_lower for hr in HIGH_RISK_CONDITIONS):
            high_risk.append({"condition": condition, "severity": "high"})
        elif any(mr in condition_lower for mr in MODERATE_RISK_CONDITIONS):
            moderate_risk.append({"condition": condition, "severity": "moderate"})
        else:
            low_risk.append({"condition": condition, "severity": "low"})
    
    # Medication risk indicators
    HIGH_RISK_MEDICATIONS = {"insulin", "chemotherapy", "immunosuppressant", "anticoagulant"}
    medication_risks = []
    for med in medications:
        med_lower = med.lower()
        if any(hrm in med_lower for hrm in HIGH_RISK_MEDICATIONS):
            medication_risks.append({"medication": med, "risk_indicator": "high"})
        else:
            medication_risks.append({"medication": med, "risk_indicator": "standard"})
    
    # Family history assessment
    family_risk_factors = []
    HIGH_RISK_FAMILY = {"heart_disease", "cancer", "diabetes", "stroke", "alzheimer"}
    for history in family_history:
        history_lower = history.lower().replace(" ", "_")
        if any(hrf in history_lower for hrf in HIGH_RISK_FAMILY):
            family_risk_factors.append({"condition": history, "hereditary_risk": "elevated"})
        else:
            family_risk_factors.append({"condition": history, "hereditary_risk": "standard"})
    
    # Lifestyle assessment
    lifestyle_risk = "unknown"
    if lifestyle_factors:
        smoking = lifestyle_factors.get("smoking", False)
        alcohol = lifestyle_factors.get("alcohol_weekly_units") or 0
        exercise = lifestyle_factors.get("exercise_weekly_hours") or 0
        
        # Ensure numeric values
        try:
            alcohol = float(alcohol) if alcohol else 0
        except (TypeError, ValueError):
            alcohol = 0
        try:
            exercise = float(exercise) if exercise else 0
        except (TypeError, ValueError):
            exercise = 0
        
        risk_score = 0
        if smoking:
            risk_score += 3
        if alcohol > 14:
            risk_score += 2
        elif alcohol > 7:
            risk_score += 1
        if exercise < 2:
            risk_score += 1
        
        lifestyle_risk = "low" if risk_score == 0 else "moderate" if risk_score <= 2 else "high"
    
    result = {
        "conditions": {
            "high_risk": high_risk,
            "moderate_risk": moderate_risk,
            "low_risk": low_risk,
            "total_count": len(medical_conditions)
        },
        "medications": {
            "items": medication_risks,
            "total_count": len(medications)
        },
        "family_history": {
            "items": family_risk_factors,
            "total_count": len(family_history)
        },
        "lifestyle_risk": lifestyle_risk,
        "overall_risk_indicators": {
            "high_risk_count": len(high_risk),
            "requires_medical_review": len(high_risk) > 0 or len([m for m in medication_risks if m["risk_indicator"] == "high"]) > 0
        }
    }
    
    return json.dumps(result, indent=2)


def evaluate_policy_rules(
    applicant_age: int,
    coverage_type: str,
    coverage_amount: float,
    health_risk_level: str,
    pre_existing_conditions: List[str],
    policy_term_years: Optional[int] = None,
) -> str:
    """
    Evaluates applicant against underwriting policy rules.
    
    :param applicant_age: Applicant's age
    :param coverage_type: Type of coverage (life, health, disability)
    :param coverage_amount: Requested coverage amount
    :param health_risk_level: Overall health risk (low, moderate, high, very_high)
    :param pre_existing_conditions: List of pre-existing conditions
    :param policy_term_years: Term length in years (for term policies)
    :return: JSON string with policy rule evaluation results
    """
    logger.info("TOOL EXECUTION: evaluate_policy_rules(age=%d, coverage=%s, amount=%.2f)", 
                applicant_age, coverage_type, coverage_amount)
    
    rule_evaluations = []
    warnings = []
    exclusions = []
    
    # Age eligibility rules
    if applicant_age < 18:
        rule_evaluations.append({
            "rule": "minimum_age",
            "passed": False,
            "message": "Applicant must be at least 18 years old"
        })
    elif applicant_age > 75:
        rule_evaluations.append({
            "rule": "maximum_age",
            "passed": False,
            "message": "Applicant exceeds maximum age limit of 75"
        })
    else:
        rule_evaluations.append({
            "rule": "age_eligibility",
            "passed": True,
            "message": f"Age {applicant_age} within eligible range"
        })
    
    # Coverage amount limits by risk level
    max_coverage = {
        "low": 5000000,
        "moderate": 2000000,
        "high": 500000,
        "very_high": 100000
    }.get(health_risk_level.lower(), 500000)
    
    if coverage_amount > max_coverage:
        rule_evaluations.append({
            "rule": "coverage_limit",
            "passed": False,
            "message": f"Requested amount ${coverage_amount:,.0f} exceeds maximum ${max_coverage:,.0f} for {health_risk_level} risk"
        })
        warnings.append(f"Coverage reduced to ${max_coverage:,.0f} based on risk assessment")
    else:
        rule_evaluations.append({
            "rule": "coverage_limit",
            "passed": True,
            "message": f"Requested amount within limits for {health_risk_level} risk level"
        })
    
    # Pre-existing condition exclusions
    EXCLUDED_CONDITIONS = {"terminal_illness", "active_cancer", "aids", "organ_transplant_pending"}
    SURCHARGE_CONDITIONS = {"diabetes", "heart_disease", "copd", "sleep_apnea"}
    
    for condition in pre_existing_conditions:
        condition_lower = condition.lower().replace(" ", "_")
        if any(ec in condition_lower for ec in EXCLUDED_CONDITIONS):
            exclusions.append({
                "condition": condition,
                "action": "decline",
                "reason": "Condition falls under automatic decline criteria"
            })
        elif any(sc in condition_lower for sc in SURCHARGE_CONDITIONS):
            warnings.append(f"Premium surcharge may apply for {condition}")
    
    if exclusions:
        rule_evaluations.append({
            "rule": "pre_existing_conditions",
            "passed": False,
            "message": f"{len(exclusions)} condition(s) require automatic decline"
        })
    elif warnings:
        rule_evaluations.append({
            "rule": "pre_existing_conditions",
            "passed": True,
            "message": f"Conditions acceptable with {len(warnings)} surcharge consideration(s)"
        })
    else:
        rule_evaluations.append({
            "rule": "pre_existing_conditions",
            "passed": True,
            "message": "No concerning pre-existing conditions"
        })
    
    # Calculate overall eligibility
    all_passed = all(r["passed"] for r in rule_evaluations)
    
    result = {
        "eligible": all_passed and len(exclusions) == 0,
        "decision": "approve" if all_passed and not exclusions else "decline" if exclusions else "review",
        "rule_evaluations": rule_evaluations,
        "warnings": warnings,
        "exclusions": exclusions,
        "recommended_coverage_amount": min(coverage_amount, max_coverage) if all_passed else 0,
        "risk_classification": health_risk_level
    }
    
    return json.dumps(result, indent=2)


def lookup_underwriting_guidelines(
    condition_name: str,
    coverage_type: str = "life",
) -> str:
    """
    Looks up underwriting guidelines for a specific condition.
    
    :param condition_name: Name of the medical condition
    :param coverage_type: Type of coverage (life, health, disability)
    :return: JSON string with underwriting guidelines
    """
    logger.info("TOOL EXECUTION: lookup_underwriting_guidelines(condition=%s, type=%s)", 
                condition_name, coverage_type)
    
    # Simplified guideline database
    GUIDELINES = {
        "diabetes": {
            "life": {
                "category": "rated",
                "table_rating": "B-D depending on control",
                "requirements": ["HbA1c results", "Current medications", "Complication history"],
                "considerations": ["Duration of diagnosis", "Type 1 vs Type 2", "Evidence of complications"]
            },
            "health": {
                "category": "covered_with_exclusion",
                "exclusion_period": "12 months",
                "requirements": ["Recent labs", "Physician statement"],
                "considerations": ["Medication compliance", "Related conditions"]
            }
        },
        "hypertension": {
            "life": {
                "category": "standard_to_rated",
                "table_rating": "Standard to Table B",
                "requirements": ["Blood pressure readings", "Current medications"],
                "considerations": ["Control level", "Duration", "End organ damage"]
            },
            "health": {
                "category": "covered",
                "exclusion_period": None,
                "requirements": ["Current medications list"],
                "considerations": ["Medication compliance"]
            }
        },
        "cancer": {
            "life": {
                "category": "postpone_or_decline",
                "table_rating": "Depends on type, stage, and time since treatment",
                "requirements": ["Pathology report", "Treatment records", "Follow-up scans"],
                "considerations": ["Cancer type", "Stage at diagnosis", "Time since remission", "Recurrence history"]
            },
            "health": {
                "category": "covered_with_exclusion",
                "exclusion_period": "24-60 months",
                "requirements": ["Complete medical records", "Oncologist statement"],
                "considerations": ["Active treatment status", "Prognosis"]
            }
        }
    }
    
    # Normalize condition name for lookup
    condition_key = condition_name.lower().replace(" ", "_")
    
    # Find matching guideline
    guideline = None
    for key, value in GUIDELINES.items():
        if key in condition_key or condition_key in key:
            guideline = value.get(coverage_type.lower(), value.get("life"))
            break
    
    if not guideline:
        guideline = {
            "category": "refer_to_underwriter",
            "table_rating": "Individual assessment required",
            "requirements": ["Complete medical records", "Physician statement"],
            "considerations": ["Full medical history review required"]
        }
    
    result = {
        "condition": condition_name,
        "coverage_type": coverage_type,
        "guideline": guideline,
        "source": "underwriting_manual_v2.1",
        "last_updated": "2024-01-15"
    }
    
    return json.dumps(result, indent=2)


def calculate_risk_score(
    health_risk_level: str,
    condition_count: int,
    age_factor: str,
    lifestyle_risk: str,
    family_history_risk: str,
) -> str:
    """
    Calculates a numerical risk score for underwriting decisions.
    
    :param health_risk_level: Overall health risk (low, moderate, high, very_high)
    :param condition_count: Number of medical conditions
    :param age_factor: Age risk factor (low, moderate, elevated, high)
    :param lifestyle_risk: Lifestyle risk level (low, moderate, high)
    :param family_history_risk: Family history risk (low, elevated)
    :return: JSON string with risk score and breakdown
    """
    logger.info("TOOL EXECUTION: calculate_risk_score(health=%s, conditions=%d)", 
                health_risk_level, condition_count)
    
    # Base score starts at 100 (lower is better)
    base_score = 100
    
    # Health risk impact
    health_impact = {
        "low": 0,
        "moderate": 50,
        "high": 150,
        "very_high": 300
    }.get(health_risk_level.lower(), 100)
    
    # Condition count impact
    condition_impact = condition_count * 20
    
    # Age factor impact
    age_impact = {
        "low": 0,
        "moderate": 30,
        "elevated": 75,
        "high": 150
    }.get(age_factor.lower(), 50)
    
    # Lifestyle impact
    lifestyle_impact = {
        "low": 0,
        "moderate": 40,
        "high": 100
    }.get(lifestyle_risk.lower(), 30)
    
    # Family history impact
    family_impact = {
        "low": 0,
        "standard": 0,
        "elevated": 50
    }.get(family_history_risk.lower(), 25)
    
    # Calculate total score
    total_score = base_score + health_impact + condition_impact + age_impact + lifestyle_impact + family_impact
    
    # Determine risk class
    if total_score <= 120:
        risk_class = "preferred_plus"
        premium_multiplier = 0.8
    elif total_score <= 150:
        risk_class = "preferred"
        premium_multiplier = 0.9
    elif total_score <= 200:
        risk_class = "standard_plus"
        premium_multiplier = 1.0
    elif total_score <= 250:
        risk_class = "standard"
        premium_multiplier = 1.1
    elif total_score <= 350:
        risk_class = "substandard"
        premium_multiplier = 1.5
    else:
        risk_class = "decline"
        premium_multiplier = None
    
    result = {
        "total_score": total_score,
        "risk_class": risk_class,
        "premium_multiplier": premium_multiplier,
        "score_breakdown": {
            "base_score": base_score,
            "health_risk_impact": health_impact,
            "condition_count_impact": condition_impact,
            "age_factor_impact": age_impact,
            "lifestyle_impact": lifestyle_impact,
            "family_history_impact": family_impact
        },
        "recommendation": "approve" if risk_class not in ["decline"] else "decline",
        "requires_medical_exam": total_score > 200
    }
    
    return json.dumps(result, indent=2)


def validate_coverage_eligibility(
    applicant_id: str,
    coverage_type: str,
    coverage_amount: float,
    policy_state: str,
    employment_status: str,
) -> str:
    """
    Validates if an applicant is eligible for requested coverage.
    
    :param applicant_id: Unique applicant identifier
    :param coverage_type: Type of coverage requested
    :param coverage_amount: Amount of coverage requested
    :param policy_state: State where policy will be issued
    :param employment_status: Current employment status
    :return: JSON string with eligibility determination
    """
    logger.info("TOOL EXECUTION: validate_coverage_eligibility(applicant=%s, type=%s)", 
                applicant_id, coverage_type)
    
    eligibility_checks = []
    
    # State availability check
    UNAVAILABLE_STATES = {"NY_group", "WA_individual"}
    state_key = f"{policy_state}_{coverage_type}"
    if state_key in UNAVAILABLE_STATES:
        eligibility_checks.append({
            "check": "state_availability",
            "passed": False,
            "message": f"Coverage type not available in {policy_state}"
        })
    else:
        eligibility_checks.append({
            "check": "state_availability",
            "passed": True,
            "message": f"Coverage available in {policy_state}"
        })
    
    # Minimum/maximum coverage amounts
    MIN_COVERAGE = {"life": 25000, "health": 0, "disability": 500}
    MAX_COVERAGE = {"life": 10000000, "health": 0, "disability": 15000}
    
    min_amt = MIN_COVERAGE.get(coverage_type.lower(), 0)
    max_amt = MAX_COVERAGE.get(coverage_type.lower(), 10000000)
    
    if coverage_amount < min_amt:
        eligibility_checks.append({
            "check": "minimum_coverage",
            "passed": False,
            "message": f"Coverage amount below minimum ${min_amt:,.0f}"
        })
    elif coverage_amount > max_amt and max_amt > 0:
        eligibility_checks.append({
            "check": "maximum_coverage",
            "passed": False,
            "message": f"Coverage amount exceeds maximum ${max_amt:,.0f}"
        })
    else:
        eligibility_checks.append({
            "check": "coverage_amount",
            "passed": True,
            "message": "Coverage amount within acceptable range"
        })
    
    # Employment status check for disability
    if coverage_type.lower() == "disability":
        if employment_status.lower() not in ["employed", "self_employed"]:
            eligibility_checks.append({
                "check": "employment_status",
                "passed": False,
                "message": "Disability coverage requires active employment"
            })
        else:
            eligibility_checks.append({
                "check": "employment_status",
                "passed": True,
                "message": "Employment status verified"
            })
    
    # Overall eligibility
    all_passed = all(c["passed"] for c in eligibility_checks)
    
    result = {
        "applicant_id": applicant_id,
        "eligible": all_passed,
        "coverage_type": coverage_type,
        "coverage_amount": coverage_amount,
        "checks": eligibility_checks,
        "next_steps": ["proceed_to_underwriting"] if all_passed else ["address_eligibility_issues"]
    }
    
    return json.dumps(result, indent=2)


def generate_decision_summary(
    applicant_name: str,
    decision: str,
    risk_class: str,
    coverage_amount: float,
    premium_estimate: Optional[float] = None,
    conditions: Optional[List[str]] = None,
    exclusions: Optional[List[str]] = None,
) -> str:
    """
    Generates a human-readable decision summary.
    
    :param applicant_name: Name of the applicant
    :param decision: Decision (approve, decline, review)
    :param risk_class: Assigned risk classification
    :param coverage_amount: Approved coverage amount
    :param premium_estimate: Estimated premium (optional)
    :param conditions: Any conditions on approval (optional)
    :param exclusions: Any exclusions (optional)
    :return: JSON string with formatted decision summary
    """
    logger.info("TOOL EXECUTION: generate_decision_summary(applicant=%s, decision=%s)", 
                applicant_name, decision)
    
    # Generate summary text based on decision
    if decision.lower() == "approve":
        summary_text = f"""
UNDERWRITING DECISION: APPROVED

Applicant: {applicant_name}
Risk Classification: {risk_class.replace('_', ' ').title()}
Approved Coverage: ${coverage_amount:,.0f}
"""
        if premium_estimate:
            summary_text += f"Estimated Annual Premium: ${premium_estimate:,.2f}\n"
        
        if conditions:
            summary_text += f"\nConditions of Approval:\n"
            for i, cond in enumerate(conditions, 1):
                summary_text += f"  {i}. {cond}\n"
        
        if exclusions:
            summary_text += f"\nPolicy Exclusions:\n"
            for i, excl in enumerate(exclusions, 1):
                summary_text += f"  {i}. {excl}\n"
                
    elif decision.lower() == "decline":
        summary_text = f"""
UNDERWRITING DECISION: DECLINED

Applicant: {applicant_name}

We regret to inform you that after careful review, we are unable to offer coverage at this time.
"""
        if exclusions:
            summary_text += f"\nReasons for Decline:\n"
            for i, excl in enumerate(exclusions, 1):
                summary_text += f"  {i}. {excl}\n"
                
    else:  # review
        summary_text = f"""
UNDERWRITING DECISION: PENDING REVIEW

Applicant: {applicant_name}
Current Risk Classification: {risk_class.replace('_', ' ').title()}
Requested Coverage: ${coverage_amount:,.0f}

This application requires additional review by our underwriting team.
"""
        if conditions:
            summary_text += f"\nAdditional Information Required:\n"
            for i, cond in enumerate(conditions, 1):
                summary_text += f"  {i}. {cond}\n"
    
    result = {
        "applicant_name": applicant_name,
        "decision": decision,
        "risk_class": risk_class,
        "coverage_amount": coverage_amount,
        "premium_estimate": premium_estimate,
        "conditions": conditions or [],
        "exclusions": exclusions or [],
        "summary_text": summary_text.strip(),
        "generated_at": datetime.now().isoformat()
    }
    
    return json.dumps(result, indent=2)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _calculate_overall_risk(risk_levels: List[Optional[str]]) -> str:
    """Calculate overall risk from multiple risk levels."""
    risk_scores = {"low": 1, "moderate": 2, "elevated": 3, "high": 4, "very_high": 5, "critical": 6}
    
    valid_risks = [r for r in risk_levels if r]
    if not valid_risks:
        return "unknown"
    
    scores = [risk_scores.get(r.lower(), 2) for r in valid_risks]
    avg_score = sum(scores) / len(scores)
    
    if avg_score <= 1.5:
        return "low"
    elif avg_score <= 2.5:
        return "moderate"
    elif avg_score <= 3.5:
        return "elevated"
    elif avg_score <= 4.5:
        return "high"
    else:
        return "very_high"


# =============================================================================
# TOOL DEFINITIONS (Schemas for Azure AI Foundry)
# =============================================================================

@dataclass
class ToolDefinition:
    """Definition of a function tool for Azure AI Foundry."""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    
    def to_foundry_schema(self) -> Dict[str, Any]:
        """Convert to Azure AI Foundry function tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


# Tool definitions registry
TOOL_DEFINITIONS: Dict[str, ToolDefinition] = {
    "analyze_health_metrics": ToolDefinition(
        name="analyze_health_metrics",
        description="Analyzes health metrics (age, height, weight, blood pressure, cholesterol, glucose) and returns a structured risk assessment with BMI calculation and categorization.",
        parameters={
            "type": "object",
            "properties": {
                "age": {"type": "integer", "description": "Applicant's age in years"},
                "height_cm": {"type": "number", "description": "Height in centimeters"},
                "weight_kg": {"type": "number", "description": "Weight in kilograms"},
                "blood_pressure_systolic": {"type": "integer", "description": "Systolic blood pressure (optional)"},
                "blood_pressure_diastolic": {"type": "integer", "description": "Diastolic blood pressure (optional)"},
                "cholesterol_total": {"type": "number", "description": "Total cholesterol in mg/dL (optional)"},
                "glucose_fasting": {"type": "number", "description": "Fasting glucose in mg/dL (optional)"}
            },
            "required": ["age", "height_cm", "weight_kg"]
        },
        function=analyze_health_metrics
    ),
    
    "extract_risk_indicators": ToolDefinition(
        name="extract_risk_indicators",
        description="Extracts and categorizes risk indicators from medical history including conditions, medications, family history, and lifestyle factors.",
        parameters={
            "type": "object",
            "properties": {
                "medical_conditions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of diagnosed medical conditions"
                },
                "medications": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of current medications"
                },
                "family_history": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of family medical history items"
                },
                "lifestyle_factors": {
                    "type": "object",
                    "description": "Dict with smoking (bool), alcohol_weekly_units (int), exercise_weekly_hours (int)",
                    "properties": {
                        "smoking": {"type": "boolean"},
                        "alcohol_weekly_units": {"type": "integer"},
                        "exercise_weekly_hours": {"type": "integer"}
                    }
                }
            },
            "required": ["medical_conditions", "medications", "family_history"]
        },
        function=extract_risk_indicators
    ),
    
    "evaluate_policy_rules": ToolDefinition(
        name="evaluate_policy_rules",
        description="Evaluates an applicant against underwriting policy rules including age limits, coverage limits, and pre-existing condition exclusions.",
        parameters={
            "type": "object",
            "properties": {
                "applicant_age": {"type": "integer", "description": "Applicant's age"},
                "coverage_type": {"type": "string", "description": "Type of coverage (life, health, disability)"},
                "coverage_amount": {"type": "number", "description": "Requested coverage amount"},
                "health_risk_level": {"type": "string", "description": "Overall health risk (low, moderate, high, very_high)"},
                "pre_existing_conditions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of pre-existing conditions"
                },
                "policy_term_years": {"type": "integer", "description": "Term length in years (optional)"}
            },
            "required": ["applicant_age", "coverage_type", "coverage_amount", "health_risk_level", "pre_existing_conditions"]
        },
        function=evaluate_policy_rules
    ),
    
    "lookup_underwriting_guidelines": ToolDefinition(
        name="lookup_underwriting_guidelines",
        description="Looks up official underwriting guidelines for a specific medical condition and coverage type.",
        parameters={
            "type": "object",
            "properties": {
                "condition_name": {"type": "string", "description": "Name of the medical condition"},
                "coverage_type": {"type": "string", "description": "Type of coverage (life, health, disability)", "default": "life"}
            },
            "required": ["condition_name"]
        },
        function=lookup_underwriting_guidelines
    ),
    
    "calculate_risk_score": ToolDefinition(
        name="calculate_risk_score",
        description="Calculates a numerical risk score for underwriting decisions based on multiple risk factors.",
        parameters={
            "type": "object",
            "properties": {
                "health_risk_level": {"type": "string", "description": "Overall health risk (low, moderate, high, very_high)"},
                "condition_count": {"type": "integer", "description": "Number of medical conditions"},
                "age_factor": {"type": "string", "description": "Age risk factor (low, moderate, elevated, high)"},
                "lifestyle_risk": {"type": "string", "description": "Lifestyle risk level (low, moderate, high)"},
                "family_history_risk": {"type": "string", "description": "Family history risk (low, elevated)"}
            },
            "required": ["health_risk_level", "condition_count", "age_factor", "lifestyle_risk", "family_history_risk"]
        },
        function=calculate_risk_score
    ),
    
    "validate_coverage_eligibility": ToolDefinition(
        name="validate_coverage_eligibility",
        description="Validates if an applicant is eligible for the requested coverage based on state, amount, and status.",
        parameters={
            "type": "object",
            "properties": {
                "applicant_id": {"type": "string", "description": "Unique applicant identifier"},
                "coverage_type": {"type": "string", "description": "Type of coverage requested"},
                "coverage_amount": {"type": "number", "description": "Amount of coverage requested"},
                "policy_state": {"type": "string", "description": "State where policy will be issued"},
                "employment_status": {"type": "string", "description": "Current employment status"}
            },
            "required": ["applicant_id", "coverage_type", "coverage_amount", "policy_state", "employment_status"]
        },
        function=validate_coverage_eligibility
    ),
    
    "generate_decision_summary": ToolDefinition(
        name="generate_decision_summary",
        description="Generates a human-readable decision summary letter for the applicant.",
        parameters={
            "type": "object",
            "properties": {
                "applicant_name": {"type": "string", "description": "Name of the applicant"},
                "decision": {"type": "string", "description": "Decision (approve, decline, review)"},
                "risk_class": {"type": "string", "description": "Assigned risk classification"},
                "coverage_amount": {"type": "number", "description": "Approved coverage amount"},
                "premium_estimate": {"type": "number", "description": "Estimated premium (optional)"},
                "conditions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Any conditions on approval"
                },
                "exclusions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Any exclusions"
                }
            },
            "required": ["applicant_name", "decision", "risk_class", "coverage_amount"]
        },
        function=generate_decision_summary
    ),
}


# =============================================================================
# AGENT TOOL REGISTRY
# Maps agent names to their available tools
# =============================================================================

AGENT_TOOLS: Dict[str, List[str]] = {
    "health_data_analysis": [
        "analyze_health_metrics",
        "extract_risk_indicators",
    ],
    "policy_risk_analysis": [
        "evaluate_policy_rules",
        "lookup_underwriting_guidelines",
        "calculate_risk_score",
    ],
    "business_rules_validation": [
        "validate_coverage_eligibility",
        "evaluate_policy_rules",
    ],
    "communication": [
        "generate_decision_summary",
    ],
}


def get_tools_for_agent(agent_name: str) -> List[ToolDefinition]:
    """Get the tool definitions for a specific agent."""
    tool_names = AGENT_TOOLS.get(agent_name, [])
    return [TOOL_DEFINITIONS[name] for name in tool_names if name in TOOL_DEFINITIONS]


def get_tool_schemas_for_agent(agent_name: str) -> List[Dict[str, Any]]:
    """Get the Foundry-compatible tool schemas for a specific agent."""
    tools = get_tools_for_agent(agent_name)
    return [tool.to_foundry_schema() for tool in tools]


def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Execute a tool by name with the given arguments.
    
    Args:
        tool_name: Name of the tool to execute
        arguments: Dictionary of arguments to pass to the tool
        
    Returns:
        JSON string result from the tool
    """
    if tool_name not in TOOL_DEFINITIONS:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    
    tool = TOOL_DEFINITIONS[tool_name]
    try:
        logger.info("Executing tool: %s with args: %s", tool_name, list(arguments.keys()))
        result = tool.function(**arguments)
        logger.info("Tool %s executed successfully", tool_name)
        return result
    except Exception as e:
        logger.error("Tool %s execution failed: %s", tool_name, e)
        return json.dumps({"error": str(e), "tool": tool_name})


def get_all_tool_functions() -> Set[Callable]:
    """Get all tool functions for use with FunctionTool."""
    return {tool.function for tool in TOOL_DEFINITIONS.values()}
