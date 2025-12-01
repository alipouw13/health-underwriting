"""
Persona definitions and configurations for WorkbenchIQ.

This module defines different industry personas (Underwriting, Claims, Mortgage)
with their specific field schemas, prompts, and analyzer configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class PersonaType(str, Enum):
    """Available persona types in WorkbenchIQ."""
    UNDERWRITING = "underwriting"
    CLAIMS = "claims"
    MORTGAGE = "mortgage"


@dataclass
class PersonaConfig:
    """Configuration for a specific persona."""
    id: str
    name: str
    description: str
    icon: str  # Emoji or icon identifier
    color: str  # Primary color for UI theming
    field_schema: Dict[str, Any]
    default_prompts: Dict[str, Dict[str, str]]
    custom_analyzer_id: str
    enabled: bool = True  # Whether this persona is fully implemented


# =============================================================================
# UNDERWRITING PERSONA CONFIGURATION
# =============================================================================

UNDERWRITING_FIELD_SCHEMA = {
    "name": "UnderwritingFields",
    "fields": {
        # ===== Personal Information =====
        "ApplicantName": {
            "type": "string",
            "description": "Full legal name of the insurance applicant, typically found at the top of the application form in Section 1 or Personal Details section. May be labeled as 'Name', 'Applicant Name', 'Proposed Insured', 'Insured Name', or 'Full Name'. Format is usually FirstName MiddleName LastName or LastName, FirstName.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "DateOfBirth": {
            "type": "date",
            "description": "Applicant's date of birth, typically found near the top of the application in the personal information section. May be labeled as 'Date of Birth', 'DOB', 'Birth Date', or 'D.O.B.'. Accepts formats like MM/DD/YYYY, DD-MM-YYYY, or Month DD, YYYY.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "Age": {
            "type": "number",
            "description": "Current age of the applicant in years, typically found in the personal details section. May be labeled as 'Age', 'Current Age', 'Age at Application', or 'Age Last Birthday'. Should be a whole number.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "Gender": {
            "type": "string",
            "description": "Biological sex or gender of the applicant, typically found in personal information section near name and date of birth. May be labeled as 'Sex', 'Gender', 'M/F', or shown as checkboxes. Common values: Male, Female, M, F, or Other.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "Nationality": {
            "type": "string",
            "description": "Nationality or citizenship of the applicant, often found in personal or residency information section. May be labeled as 'Nationality', 'Citizenship', 'Country of Citizenship', or 'Citizen of'. Extract the country name.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "Residency": {
            "type": "string",
            "description": "Current country, province/state, or city of residence, typically found in contact information or address section. May be labeled as 'Residence', 'Country of Residence', 'Residential Address', 'Province/State', or 'Place of Residence'. Include city and country/state if available.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "Occupation": {
            "type": "string",
            "description": "Current occupation, job title, or profession of the applicant, typically found in occupation/employment section. May be labeled as 'Occupation', 'Profession', 'Employment', 'Job Title', 'Current Position', or 'Nature of Work'. Extract the specific job title or description.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "Height": {
            "type": "string",
            "description": "Height of the applicant, typically found in physical examination or medical information section. May be labeled as 'Height', 'Ht.', or 'Stature'. Include units such as feet/inches (e.g., 5'10\"), centimeters (e.g., 178 cm), or meters.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "Weight": {
            "type": "string",
            "description": "Body weight of the applicant, typically found in physical examination section near height. May be labeled as 'Weight', 'Wt.', 'Body Weight', or 'Current Weight'. Include units such as pounds (lb), kilograms (kg), or lbs.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        
        # ===== Policy Information =====
        "PolicyProduct": {
            "type": "string",
            "description": "Name or type of the insurance product being applied for, typically found at the top of the application or in product selection section. May be labeled as 'Policy Type', 'Product Name', 'Plan Name', 'Insurance Product', 'Coverage Type', or 'Policy Applied For'. Examples: Term Life, Whole Life, Universal Life, Critical Illness.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "CoverageAmount": {
            "type": "string",
            "description": "Total sum insured or face amount of coverage requested, typically found in policy details section near product name. May be labeled as 'Coverage Amount', 'Sum Insured', 'Face Amount', 'Death Benefit', 'Amount of Insurance', or 'Coverage'. Include currency symbol (e.g., $500,000, CAD 1,000,000).",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "PremiumAmount": {
            "type": "string",
            "description": "Premium amount to be paid, typically found in policy details or premium calculation section. May be labeled as 'Premium', 'Monthly Premium', 'Annual Premium', 'Premium Amount', or 'Payment Amount'. Include currency and payment frequency (e.g., $150/month, $1,800 annually).",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "RatingClass": {
            "type": "string",
            "description": "Underwriting risk classification or rating class assigned, typically found in underwriting decision or rating section. May be labeled as 'Rating', 'Risk Class', 'Rating Class', 'Classification', 'Underwriting Class', or 'Health Rating'. Common values: Preferred Plus, Preferred, Standard Plus, Standard, Substandard, Table Rating (A-J).",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        
        # ===== Family History =====
        "FamilyHistory": {
            "type": "array",
            "description": "Family medical history information, typically found in family history section or medical questionnaire. May be labeled as 'Family History', 'Family Medical History', 'History of Parents/Siblings', or 'Hereditary Conditions'.",
            "items": {
                "type": "object",
                "properties": {
                    "relationship": {
                        "type": "string",
                        "description": "Relationship to applicant (Father, Mother, Brother, Sister, Paternal Grandfather, etc.)"
                    },
                    "condition": {
                        "type": "string",
                        "description": "Medical condition or cause of death (Heart Disease, Cancer, Diabetes, Stroke, etc.)"
                    },
                    "ageAtDiagnosis": {
                        "type": "string",
                        "description": "Age when condition was diagnosed or age at death"
                    },
                    "livingStatus": {
                        "type": "string",
                        "description": "Living or Deceased"
                    }
                }
            },
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "FamilyHistorySummary": {
            "type": "string",
            "description": "Summary of family medical history in text format, found in section 2 (Family history) or family history questionnaire. Include for each family member: relationship (Father, Mother, Brother, Sister), current age or age at death, state of health or cause of death, and any hereditary conditions (heart disease, cancer, diabetes, kidney disease, mental illness, neurological conditions). Format as semicolon-separated entries like 'Father: Age at death 65, Cause: heart disease; Mother: Age 70, Living, Healthy'.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        
        # ===== Blood Pressure =====
        "BloodPressureReadings": {
            "type": "array",
            "description": "Blood pressure measurements, typically found in examination results or vital signs section. May be labeled as 'Blood Pressure', 'BP', 'B/P', 'Systolic/Diastolic', or 'Pressure'.",
            "items": {
                "type": "object",
                "properties": {
                    "systolic": {
                        "type": "number",
                        "description": "Systolic blood pressure (top number, typically 90-140)"
                    },
                    "diastolic": {
                        "type": "number",
                        "description": "Diastolic blood pressure (bottom number, typically 60-90)"
                    },
                    "date": {
                        "type": "string",
                        "description": "Date and/or time of reading if available"
                    },
                    "position": {
                        "type": "string",
                        "description": "Position during reading (Sitting, Standing, Supine) if specified"
                    }
                }
            },
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "HypertensionDiagnosis": {
            "type": "string",
            "description": "Diagnosis or history of high blood pressure/hypertension, typically found in medical history section. May be labeled as 'Hypertension', 'High Blood Pressure', 'HTN', 'Elevated BP', or 'Blood Pressure Condition'. Include diagnosis date, duration (e.g., 'diagnosed 2019'), treatment details, and medication names if mentioned.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        
        # ===== Cholesterol / Lipid Panel =====
        "LipidPanelResults": {
            "type": "object",
            "description": "Lipid panel or cholesterol test results, typically found in laboratory results or blood test section. May be labeled as 'Lipid Profile', 'Cholesterol Panel', 'Lipid Test', 'Blood Lipids', or 'Cholesterol Results'.",
            "properties": {
                "totalCholesterol": {
                    "type": "string",
                    "description": "Total cholesterol value with units (e.g., 200 mg/dL, 5.2 mmol/L)"
                },
                "ldl": {
                    "type": "string",
                    "description": "LDL (bad cholesterol) value with units. May be labeled 'LDL', 'LDL-C', or 'Low Density Lipoprotein'"
                },
                "hdl": {
                    "type": "string",
                    "description": "HDL (good cholesterol) value with units. May be labeled 'HDL', 'HDL-C', or 'High Density Lipoprotein'"
                },
                "triglycerides": {
                    "type": "string",
                    "description": "Triglycerides value with units. May be labeled 'Triglycerides', 'TG', or 'TRIG'"
                },
                "testDate": {
                    "type": "string",
                    "description": "Date when lipid panel was performed"
                }
            },
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "CholesterolMedications": {
            "type": "array",
            "description": "Medications for cholesterol management, typically found in current medications or treatment section. May be labeled as 'Cholesterol Medications', 'Lipid-lowering drugs', 'Statins', or within general medication list.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Medication name (e.g., Atorvastatin, Lipitor, Crestor, Simvastatin)"
                    },
                    "dosage": {
                        "type": "string",
                        "description": "Dosage strength and frequency (e.g., 20mg daily, 40mg once daily)"
                    }
                }
            },
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        
        # ===== Current Medications =====
        "CurrentMedications": {
            "type": "array",
            "description": "All current medications being taken by the applicant, typically found in medications section or medical history. May be labeled as 'Current Medications', 'Prescriptions', 'Medication List', 'Drugs Currently Taking', or 'Present Medications'.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Medication name (brand or generic)"
                    },
                    "dosage": {
                        "type": "string",
                        "description": "Dosage strength and frequency (e.g., 10mg twice daily, 50mg as needed)"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Medical condition or reason for taking (e.g., high blood pressure, diabetes, pain)"
                    },
                    "startDate": {
                        "type": "string",
                        "description": "When medication was started, if available"
                    }
                }
            },
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "CurrentMedicationsList": {
            "type": "string",
            "description": "Summary list of all current medications in text format, found in medication sections, treatment records, or question 7. Include for each medication: name (brand or generic), dosage, frequency, and condition being treated. Format as semicolon-separated entries like 'Metformin 500mg twice daily for diabetes; Lisinopril 10mg daily for hypertension'. May be labeled as 'Current Medications', 'Prescriptions', 'Medication List', or 'Drugs Currently Taking'.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        
        # ===== Medical Conditions Summary =====
        "MedicalConditionsSummary": {
            "type": "string",
            "description": "Summary of all disclosed medical conditions, diagnoses, and health issues throughout the document. Extract information from medical history section (question 3), physician statements, examination findings, and questionnaires. Include: condition name, diagnosis date or duration, current status (resolved/ongoing/chronic), treatments received, and outcomes. Format as semicolon-separated entries. May be labeled as 'Medical History', 'Health Conditions', 'Illnesses', 'Diagnoses', or found in affirmative answers section.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        
        # ===== Diagnostic Tests =====
        "DiagnosticTests": {
            "type": "array",
            "description": "Diagnostic procedures and tests performed, typically found in medical history, test results, or examination sections. May be labeled as 'Tests', 'Diagnostic Procedures', 'Investigations', 'Medical Tests', or 'Examinations'.",
            "items": {
                "type": "object",
                "properties": {
                    "testType": {
                        "type": "string",
                        "description": "Type of test (ECG/EKG, Echocardiogram, Stress Test, X-Ray, CT Scan, MRI, Ultrasound, Blood Test, Biopsy)"
                    },
                    "date": {
                        "type": "string",
                        "description": "Date when test was performed"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Medical reason or symptoms prompting the test"
                    },
                    "result": {
                        "type": "string",
                        "description": "Test outcome or findings (Normal, Abnormal, specific findings)"
                    }
                }
            },
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "DiagnosticTestsSummary": {
            "type": "string",
            "description": "Summary of diagnostic procedures and tests performed, found in section 21 (Diagnostic Tests), question 6c, or scattered in medical history. Include: test type (ECG/EKG, Echocardiogram, Mammogram, Ultrasound, X-Ray, CT Scan, MRI, Blood Test), date performed, reason/indication, and result/finding. Format as semicolon-separated entries. May be labeled as 'Tests', 'Diagnostic Procedures', 'Investigations', or 'Medical Tests'.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        
        # ===== Surgeries and Hospitalizations =====
        "SurgeriesAndHospitalizations": {
            "type": "array",
            "description": "Past surgical procedures and hospital admissions, typically found in medical history or surgical history section. May be labeled as 'Surgeries', 'Operations', 'Surgical History', 'Hospitalizations', 'Hospital Admissions', or 'Procedures'.",
            "items": {
                "type": "object",
                "properties": {
                    "procedure": {
                        "type": "string",
                        "description": "Name of surgery or reason for hospitalization"
                    },
                    "date": {
                        "type": "string",
                        "description": "Date or year of procedure/admission"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Medical reason or diagnosis requiring procedure"
                    },
                    "outcome": {
                        "type": "string",
                        "description": "Result or recovery status (Fully recovered, Complications, Ongoing treatment)"
                    }
                }
            },
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        
        # ===== Lifestyle - Tobacco =====
        "SmokingStatus": {
            "type": "string",
            "description": "Tobacco and smoking history, typically found in lifestyle habits or tobacco use section. May be labeled as 'Smoking', 'Tobacco Use', 'Cigarette Use', 'Smoker', or 'Nicotine Use'. Indicate status: Non-smoker (never smoked), Ex-smoker/Former smoker (include quit date if available, e.g., 'quit 2020'), or Current smoker (include type such as cigarettes/cigars/pipe/chewing tobacco and quantity per day/week).",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        
        # ===== Lifestyle - Alcohol =====
        "AlcoholUse": {
            "type": "string",
            "description": "Alcohol consumption habits, typically found in lifestyle or substance use section. May be labeled as 'Alcohol Use', 'Alcohol Consumption', 'Drinking Habits', 'Alcoholic Beverages', or 'Social Drinking'. Include frequency (daily/weekly/monthly/occasional/none), type of beverage (beer/wine/spirits), and quantity (drinks per day/week, e.g., '2 beers per week', '1 glass wine daily').",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        
        # ===== Lifestyle - Drug Use =====
        "DrugUse": {
            "type": "string",
            "description": "History of recreational or illegal drug use, typically found in lifestyle or substance abuse section. May be labeled as 'Drug Use', 'Substance Abuse', 'Recreational Drugs', 'Narcotics Use', or 'Controlled Substances'. Include type of substance, frequency, dates of use, and whether past or current. If none, may state 'No' or 'None'.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        
        # ===== Existing Policies =====
        "ExistingPoliciesSummary": {
            "type": "string",
            "description": "Summary of all existing life insurance policies with the same or other insurers, found in existing insurance section and replacement/disclosure forms. May be labeled as 'Existing Insurance', 'Current Policies', 'Insurance in Force', or 'Other Coverage'. Include: insurance company name, policy type/product, coverage amount/face value, policy status (active/lapsed/terminated), issue date, and any claims history or lapses. Format as semicolon-separated entries.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        
        # ===== Physician Information =====
        "FamilyPhysician": {
            "type": "string",
            "description": "Primary care physician or family doctor information, typically found in physician information or medical attendant section. May be labeled as 'Family Doctor', 'Primary Physician', 'Personal Physician', 'Attending Physician', or 'Regular Doctor'. Include: full name (Dr. FirstName LastName), clinic/practice name, phone number, complete address (street, city, state/province), date of last visit, and reason for last visit if mentioned.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        
        # ===== Administrative Fields =====
        "ApplicationDate": {
            "type": "date",
            "description": "Date when the application was signed, submitted, or completed, typically found at bottom of application near signature section or in administrative header. May be labeled as 'Application Date', 'Date of Application', 'Date Signed', 'Signature Date', or 'Completion Date'. Accepts formats MM/DD/YYYY, DD-MM-YYYY, or written dates.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "ExaminationDate": {
            "type": "date",
            "description": "Date when the medical examination or paramedical exam was conducted, typically found at top or bottom of examination report or medical form. May be labeled as 'Exam Date', 'Examination Date', 'Date of Examination', 'Test Date', or 'Physical Date'.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "ExaminerName": {
            "type": "string",
            "description": "Name of medical examiner, paramedic, or nurse who conducted the physical examination, typically found on examination report near signature or at bottom of form. May be labeled as 'Examiner', 'Examiner Name', 'Paramedic', 'Medical Examiner', 'Examined By', or 'Nurse Name'. Extract full name.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "AgentName": {
            "type": "string",
            "description": "Name of the insurance agent, broker, or financial advisor handling the application, typically found on first page or in agent information section. May be labeled as 'Agent', 'Agent Name', 'Broker', 'Financial Advisor', 'Writing Agent', 'Producer', or 'Representative'. Include full name and agent code/number if present.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        
        # ===== Urinalysis Results =====
        "UrinalysisResults": {
            "type": "object",
            "description": "Urinalysis or urine test results, typically found in laboratory results or examination findings section. May be labeled as 'Urinalysis', 'Urine Test', 'U/A', 'Urine Screen', or 'Urine Analysis'.",
            "properties": {
                "protein": {
                    "type": "string",
                    "description": "Protein level in urine. May show as Negative, Trace, 1+, 2+, 3+, or specific value"
                },
                "glucose": {
                    "type": "string",
                    "description": "Glucose/sugar level in urine. May be labeled 'Glucose', 'Sugar', or 'GLU'. Values: Negative, Trace, or positive findings"
                },
                "blood": {
                    "type": "string",
                    "description": "Blood in urine (hematuria). Values: Negative, Trace, Small, Moderate, Large, or specific value"
                },
                "other": {
                    "type": "string",
                    "description": "Any other urinalysis findings (pH, specific gravity, ketones, leukocytes, etc.)"
                }
            },
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        
        # ===== Pulse =====
        "PulseRate": {
            "type": "number",
            "description": "Resting heart rate or pulse measurement in beats per minute, typically found in vital signs or physical examination section. May be labeled as 'Pulse', 'Heart Rate', 'HR', 'Pulse Rate', or 'Beats Per Minute'. Should be a whole number typically between 40-120 bpm.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        
        # ===== Risk Factors =====
        "ForeignTravelPlans": {
            "type": "string",
            "description": "Plans for international travel or residence outside home country, typically found in lifestyle, travel, or risk factors section. May be labeled as 'Foreign Travel', 'International Travel', 'Travel Plans', 'Residence Abroad', or 'Extended Travel'. Include destination countries, duration (number of months), purpose (business/leisure/residence), and planned dates. Focus on trips exceeding 2 months or permanent relocation.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "HazardousActivities": {
            "type": "array",
            "description": "Participation in high-risk sports or hazardous activities, typically found in avocations, hobbies, or risk activities section. May be labeled as 'Hazardous Activities', 'Aviation', 'Sports', 'Hobbies', 'Dangerous Activities', or 'High-Risk Pursuits'.",
            "items": {
                "type": "object",
                "properties": {
                    "activity": {
                        "type": "string",
                        "description": "Name of activity (Scuba Diving, Skydiving, Rock Climbing, Racing, Flying, Mountaineering, etc.)"
                    },
                    "frequency": {
                        "type": "string",
                        "description": "How often engaged in (times per year, monthly, weekly)"
                    },
                    "details": {
                        "type": "string",
                        "description": "Additional details like certifications, safety equipment, professional vs amateur"
                    }
                }
            },
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "DrivingViolations": {
            "type": "string",
            "description": "Motor vehicle violations, accidents, or license suspensions within past 5 years, typically found in driving history or motor vehicle section. May be labeled as 'Driving Record', 'Traffic Violations', 'MVR', 'Motor Vehicle History', 'License Suspensions', or 'DUI/DWI'. Include type of violation (speeding, reckless driving, DUI/DWI), date, and any license suspensions or revocations.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "CriminalRecord": {
            "type": "string",
            "description": "Criminal charges, convictions, or pending legal matters, typically found in legal history or background section. May be labeled as 'Criminal Record', 'Criminal History', 'Convictions', 'Legal History', or 'Charges'. Include nature of charge/conviction, date, and disposition (convicted, pending, dismissed).",
            "method": "extract",
            "estimateSourceAndConfidence": True
        }
    }
}

UNDERWRITING_DEFAULT_PROMPTS = {
    "application_summary": {
        "customer_profile": """
You are an expert life insurance underwriter.

Given the full underwriting application converted to Markdown, extract a concise, factual view of the customer profile and application details.

Focus on:
- Name, age, gender, smoking status
- Nationality and residency
- Occupation if present
- Policy illustration (product, coverage amount, premium, rating class)
- Any automated initial decision (risk type, reason, action)

Return STRICT JSON with this exact shape:

{
  "summary": "2–4 sentence narrative of the overall profile.",
  "key_fields": [
    {"label": "Age", "value": "48"},
    {"label": "Smoking status", "value": "Non-smoker"}
  ],
  "risk_assessment": "Low | Moderate | High",
  "underwriting_action": "Free text recommendation for next steps."
}
        """,
        "existing_policies": """
You are an expert life insurance underwriter.

Given the underwriting application in Markdown, summarise all existing policies with the same insurer.

For each existing policy capture:
- Product name
- Sum insured and currency
- Effective date
- Medical rating and exclusions, if any
- Claim history (dates, diagnoses, amounts)
- Whether the policy is active, lapsed, or surrendered

Return STRICT JSON:

{
  "summary": "Short narrative of current in-force cover and history.",
  "policies": [
    {
      "product": "Heirloom (II)",
      "status": "In force",
      "sum_insured": "SGD 500k",
      "effective_date": "2016-10-27",
      "medical_rating": "Standard",
      "exclusions": "MCC excl. severe asthma",
      "claims_summary": "e.g. rib fracture 2022, amount 6,729, claim ratio 1440%"
    }
  ],
  "total_cover_summary": "Short text describing overall cover across policies.",
  "risk_assessment": "Low | Moderate | High",
  "underwriting_action": "Free text recommendation (e.g. accept, load, or request more info)."
}
        """
    },
    "medical_summary": {
        "family_history": """
You are an expert life insurance medical underwriter.

From the Markdown document, extract details on FAMILY MEDICAL HISTORY only.

Look for:
- Conditions in first-degree relatives (e.g. heart disease, cancer, stroke)
- Age at onset and age at death when available
- Relationship to the proposed insured
- Any mention that full details are not available

Return STRICT JSON:

{
  "summary": "2–3 sentence overview of family history relevance.",
  "relatives": [
    {
      "relationship": "Father",
      "condition": "Heart disease",
      "age_at_onset": "Unknown or number",
      "age_at_death": "Over 80",
      "notes": "Any additional comments."
    }
  ],
  "risk_assessment": "Low | Moderate | High",
  "underwriting_action": "Comment on whether further clarification is needed."
}
        """,
        "hypertension": """
You are an expert life insurance medical underwriter.

Using the Markdown, focus ONLY on hypertension / raised blood pressure.

Capture:
- Diagnosis details and duration, if present
- Most recent 3–4 blood pressure readings with dates
- Any medication history (drug, dose, compliance)
- Relevant lab results (lipids, renal function) that affect HTN risk
- Previous underwriting notes or decisions related to blood pressure

Return STRICT JSON:

{
  "summary": "3–5 sentence clinical underwriting summary for hypertension.",
  "bp_readings": [
    {"date": "2024-03-15", "systolic": 134, "diastolic": 81},
    {"date": "2024-03-15", "systolic": 123, "diastolic": 76}
  ],
  "medications": [
    {"name": "Drug name", "dose": "5 mg od", "status": "Current | Past | Never"}
  ],
  "labs": [
    {"name": "Glucose (fasting)", "value": "83 mg/dL", "comment": "Within reference range"}
  ],
  "risk_assessment": "Low | Moderate | High",
  "underwriting_action": "Recommendation (e.g. standard, mild loading, defer, request APS)."
}
        """,
        "high_cholesterol": """
You are an expert life insurance medical underwriter.

Using the Markdown, focus ONLY on dyslipidaemia / hyperlipidaemia / high cholesterol.

Capture:
- Diagnosis details and duration, if available
- Latest lipid profile (Total, LDL, HDL, Triglycerides) with dates and reference ranges
- Any treatment (statins or other agents)
- Previous underwriting decisions linked to cholesterol

Return STRICT JSON:

{
  "summary": "3–4 sentence overview of cholesterol control and trend.",
  "lipid_panels": [
    {
      "date": "2024-03-15",
      "total_cholesterol": "254 mg/dL (high)",
      "ldl": "169 mg/dL",
      "hdl": "64 mg/dL",
      "triglycerides": "109 mg/dL"
    }
  ],
  "medications": [
    {"name": "Statin", "dose": "10 mg nocte", "status": "Current | Past | Never"}
  ],
  "risk_assessment": "Low | Moderate | High",
  "underwriting_action": "Recommendation regarding terms or further information."
}
        """,
        "other_medical_findings": """
You are an expert life insurance medical underwriter.

Using the Markdown, summarise OTHER MEDICAL FINDINGS not already captured in hypertension or cholesterol.

Include:
- Past medical history (e.g. reflux, slipped disc, asthma, surgeries)
- Smoking, alcohol use, and any substance use
- Any investigations (endoscopy, colonoscopy) and their outcomes
- Negative findings that are reassuring (e.g. no AIDS diagnosis)

Return STRICT JSON:

{
  "summary": "High-level narrative of other relevant medical history.",
  "conditions": [
    {
      "name": "Reflux",
      "onset": "2007",
      "status": "Resolved | Ongoing",
      "details": "Omeprazole in 2007, now recovered."
    }
  ],
  "lifestyle": {
    "smoking_status": "Non-smoker | Ex-smoker | Smoker",
    "alcohol": "Description if available",
    "other": "Any other lifestyle notes."
  },
  "risk_assessment": "Low | Moderate | High",
  "underwriting_action": "Notes on whether any additional evidence is required."
}
        """,
        "other_risks": """
You are an expert life insurance new business underwriter.

Using the Markdown, list any ADMINISTRATIVE or NON-MEDICAL RISKS and FINDINGS, such as:
- Discrepancies in signatures or dates
- Missing pages or incomplete answers
- Suspicious information requiring clarification

Return STRICT JSON:

{
  "summary": "Short description of admin / non-medical concerns.",
  "issues": [
    {
      "type": "Signature mismatch",
      "detail": "Signature on page 6 of medical exam form differs from page 20 of application.",
      "recommended_follow_up": "Request clarification from adviser and client."
    }
  ],
  "risk_assessment": "Low | Moderate | High",
  "underwriting_action": "Suggested operational next steps."
}
        """
    },
    "requirements": {
        "requirements_summary": """
You are an expert life insurance underwriter.

From the Markdown application, extract any REQUIREMENTS or PENDING ITEMS for underwriting, such as:
- Attending physician statements (APS) and for what conditions
- Additional forms or questionnaires
- Financial and AML documentation
- Any system-suggested requirements

Return STRICT JSON:

{
  "summary": "Short narrative of the requirement set.",
  "requirements": [
    {"type": "APS", "detail": "APS for slipped disc"},
    {"type": "APS", "detail": "APS for irregular / delayed period"},
    {"type": "Financial", "detail": "Proof of income and net worth"}
  ],
  "priority": "Low | Medium | High",
  "underwriting_action": "Guidance on sequencing and what to chase first."
}
        """
    }
}


# =============================================================================
# CLAIMS PERSONA CONFIGURATION (Mock/Placeholder)
# =============================================================================

CLAIMS_FIELD_SCHEMA = {
    "name": "ClaimsFields",
    "fields": {
        # ===== Claimant Information =====
        "ClaimantName": {
            "type": "string",
            "description": "Full name of the claimant or insured party filing the claim.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "PolicyNumber": {
            "type": "string",
            "description": "Insurance policy number associated with the claim.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "ClaimNumber": {
            "type": "string",
            "description": "Unique claim reference number assigned to this claim.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "DateOfLoss": {
            "type": "date",
            "description": "Date when the incident or loss occurred.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "DateOfClaim": {
            "type": "date",
            "description": "Date when the claim was filed or submitted.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        
        # ===== Medical Information =====
        "DiagnosisCodes": {
            "type": "array",
            "description": "ICD-10 or other diagnosis codes from medical records.",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "ICD-10 code"},
                    "description": {"type": "string", "description": "Description of diagnosis"}
                }
            },
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "TreatingPhysician": {
            "type": "string",
            "description": "Name and credentials of the treating physician.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "HospitalName": {
            "type": "string",
            "description": "Name of hospital or medical facility where treatment was provided.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "AdmissionDate": {
            "type": "date",
            "description": "Date of hospital admission if applicable.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "DischargeDate": {
            "type": "date",
            "description": "Date of hospital discharge if applicable.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "ProceduresConducted": {
            "type": "array",
            "description": "List of medical procedures performed.",
            "items": {
                "type": "object",
                "properties": {
                    "procedure": {"type": "string", "description": "Procedure name or CPT code"},
                    "date": {"type": "string", "description": "Date of procedure"},
                    "outcome": {"type": "string", "description": "Outcome or result"}
                }
            },
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        
        # ===== Claim Details =====
        "ClaimType": {
            "type": "string",
            "description": "Type of claim (e.g., Medical, Disability, Life, Critical Illness).",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "ClaimAmount": {
            "type": "string",
            "description": "Total amount claimed with currency.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "ClaimStatus": {
            "type": "string",
            "description": "Current status of the claim (Pending, Approved, Denied, Under Review).",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "PreExistingConditions": {
            "type": "string",
            "description": "Any pre-existing conditions that may affect the claim.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "CauseOfClaim": {
            "type": "string",
            "description": "Primary cause or reason for the claim.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        }
    }
}

CLAIMS_DEFAULT_PROMPTS = {
    "claim_summary": {
        "claim_overview": """
You are an expert insurance claims processor.

Given the claim documents converted to Markdown, extract a comprehensive overview of the claim.

Focus on:
- Claimant information and policy details
- Date of loss/incident and claim filing date
- Type of claim and amount requested
- Current claim status

Return STRICT JSON with this exact shape:

{
  "summary": "2–4 sentence narrative of the claim.",
  "key_fields": [
    {"label": "Claim Type", "value": "Medical"},
    {"label": "Claim Amount", "value": "$15,000"}
  ],
  "claim_status": "Pending | Under Review | Approved | Denied",
  "processing_action": "Free text recommendation for next steps."
}
        """,
        "medical_review": """
You are an expert medical claims reviewer.

From the claim documents, extract all relevant medical information.

Capture:
- Diagnosis codes and descriptions
- Treating physicians and facilities
- Procedures performed
- Treatment timeline

Return STRICT JSON:

{
  "summary": "Clinical summary of the medical claim.",
  "diagnoses": [
    {"code": "ICD-10 code", "description": "Diagnosis description"}
  ],
  "treatments": [
    {"procedure": "Procedure name", "date": "Date", "provider": "Provider name"}
  ],
  "medical_necessity": "Assessment of medical necessity",
  "processing_action": "Recommendation for claim processing."
}
        """
    },
    "eligibility": {
        "coverage_verification": """
You are an expert insurance claims adjuster.

Review the claim documents and verify coverage eligibility.

Check:
- Policy effective dates vs date of loss
- Coverage limits and deductibles
- Pre-existing condition exclusions
- Waiting periods

Return STRICT JSON:

{
  "summary": "Coverage eligibility assessment.",
  "coverage_status": "Eligible | Partially Eligible | Not Eligible",
  "issues": [
    {"type": "Issue type", "detail": "Description of issue"}
  ],
  "processing_action": "Recommended action."
}
        """
    }
}


# =============================================================================
# MORTGAGE PERSONA CONFIGURATION (Stub)
# =============================================================================

MORTGAGE_FIELD_SCHEMA = {
    "name": "MortgageFields",
    "fields": {
        "BorrowerName": {
            "type": "string",
            "description": "Full name of the mortgage applicant/borrower.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "PropertyAddress": {
            "type": "string",
            "description": "Address of the property being financed.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "LoanAmount": {
            "type": "string",
            "description": "Requested mortgage loan amount.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "PropertyValue": {
            "type": "string",
            "description": "Appraised value of the property.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "AnnualIncome": {
            "type": "string",
            "description": "Borrower's annual income.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "EmploymentStatus": {
            "type": "string",
            "description": "Current employment status of the borrower.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "CreditScore": {
            "type": "number",
            "description": "Borrower's credit score.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        },
        "DebtToIncomeRatio": {
            "type": "string",
            "description": "Calculated debt-to-income ratio.",
            "method": "extract",
            "estimateSourceAndConfidence": True
        }
    }
}

MORTGAGE_DEFAULT_PROMPTS = {
    "application_summary": {
        "borrower_profile": """
You are an expert mortgage underwriter.

Given the mortgage application documents, extract a comprehensive borrower profile.

Return STRICT JSON with borrower details, income verification, and initial assessment.
        """
    }
}


# =============================================================================
# PERSONA REGISTRY
# =============================================================================

PERSONA_CONFIGS: Dict[PersonaType, PersonaConfig] = {
    PersonaType.UNDERWRITING: PersonaConfig(
        id="underwriting",
        name="Underwriting",
        description="Life insurance underwriting workbench for processing applications and medical documents",
        icon="📋",
        color="#6366f1",  # Indigo
        field_schema=UNDERWRITING_FIELD_SCHEMA,
        default_prompts=UNDERWRITING_DEFAULT_PROMPTS,
        custom_analyzer_id="underwritingAnalyzer",
        enabled=True,
    ),
    PersonaType.CLAIMS: PersonaConfig(
        id="claims",
        name="Claims",
        description="Insurance claims processing workbench for reviewing medical claims and documentation",
        icon="🏥",
        color="#0891b2",  # Cyan
        field_schema=CLAIMS_FIELD_SCHEMA,
        default_prompts=CLAIMS_DEFAULT_PROMPTS,
        custom_analyzer_id="claimsAnalyzer",
        enabled=True,  # Mock enabled for demo
    ),
    PersonaType.MORTGAGE: PersonaConfig(
        id="mortgage",
        name="Mortgage",
        description="Mortgage underwriting workbench for loan applications and property documents",
        icon="🏠",
        color="#059669",  # Emerald
        field_schema=MORTGAGE_FIELD_SCHEMA,
        default_prompts=MORTGAGE_DEFAULT_PROMPTS,
        custom_analyzer_id="mortgageAnalyzer",
        enabled=False,  # Coming soon
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_persona_config(persona_id: str) -> PersonaConfig:
    """Get configuration for a specific persona by ID."""
    try:
        persona_type = PersonaType(persona_id.lower())
        return PERSONA_CONFIGS[persona_type]
    except (ValueError, KeyError):
        raise ValueError(f"Unknown persona: {persona_id}. Valid options: {[p.value for p in PersonaType]}")


def list_personas() -> List[Dict[str, Any]]:
    """List all available personas with their metadata."""
    return [
        {
            "id": config.id,
            "name": config.name,
            "description": config.description,
            "icon": config.icon,
            "color": config.color,
            "enabled": config.enabled,
        }
        for config in PERSONA_CONFIGS.values()
    ]


def get_field_schema(persona_id: str) -> Dict[str, Any]:
    """Get the field extraction schema for a persona."""
    config = get_persona_config(persona_id)
    return config.field_schema


def get_default_prompts(persona_id: str) -> Dict[str, Dict[str, str]]:
    """Get the default prompts for a persona."""
    config = get_persona_config(persona_id)
    return config.default_prompts


def get_custom_analyzer_id(persona_id: str) -> str:
    """Get the custom analyzer ID for a persona."""
    config = get_persona_config(persona_id)
    return config.custom_analyzer_id
