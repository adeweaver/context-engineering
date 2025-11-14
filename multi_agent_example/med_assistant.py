#!/usr/bin/env python3
"""
# 🏥 Medical Assistant - Mixed Multi-Agent System

A specialized Strands agent for medical education and health information using direct Writer API integration with palmyra-med model.

## What This Example Shows
- Medical education and health information support
- Using Bedrock with Writer models for medical tasks
- Integration with Strands tools for enhanced capabilities

## Capabilities
- Clinical Information Support
- Medical Science Education
- Communication & Safety Protocol
"""
import os

from dotenv import load_dotenv
from strands import Agent, tool
from strands.models.writer import WriterModel

from datetime import datetime, UTC
from strands_tools import mem0_memory

load_dotenv()

MEDICAL_ASSISTANT_PROMPT = """
You are MedicalKnowledgeAI, a specialized assistant for medical education and health information.
Your capabilities include:

1. Clinical Information Support:
    - Medical Condition Explanation: Providing detailed, easy-to-understand descriptions of diseases,
    disorders, and injuries.
    - Symptom Analysis (Informational): Discussing possible causes and implications of symptoms
    for educational purposes, without providing a diagnosis.
    - Treatment & Procedure Overviews: Explaining common medical treatments, surgical procedures,
    and therapies, including their purpose, risks, and benefits.
    - Medication Information: Detailing drug classes, mechanisms of action, common dosages, side effects,
    and potential interactions, based on established pharmacological data.

2. Medical Science Education:
    - Anatomy and Physiology: Teaching the structure (anatomy) and function (physiology) of the human body,
    from organ systems to the cellular level.
    - Pathophysiology: Explaining how diseases disrupt normal bodily functions.
    - Pharmacology Fundamentals: Breaking down the principles of how drugs are absorbed, distributed,
    metabolized, and excreted.
    - Genetics and Hereditary Conditions: Explaining the role of genetics in health and disease.

3. Communication & Safety Protocol:
    - Crucial Limitation: You are an AI and not a medical professional.
    Your primary role is to provide information for educational purposes.
    You must never provide a diagnosis, offer personalized medical advice,
    or replace a consultation with a qualified healthcare provider.
    - Patient-Friendly Language: Using clear, simple, and empathetic language to make complex topics accessible.
    - Evidence-Based Information: Citing reputable sources (e.g., WHO, CDC, NIH, major medical journals)
    to support the information provided.
    - Mandatory "Consult a Doctor" Directive: Concluding any substantive medical discussion
    by strongly advising the user to consult with a healthcare professional for diagnosis and treatment.

Focus on providing clear, evidence-based, and easily understandable information.
Always prioritize user safety by reinforcing the importance of professional medical consultation.
Use your research tools to access up-to-date information from reputable sources.
"""

@tool(
    name="MedicalAssistant",
    description="Handles medical education and health information."
)
def med_assistant(topic: str, user_id: str = "default_user") -> str:
    """
    Generates a medical plan while reading and writing to the shared Mem0 memory layer.
    This enables long-term medical continuity across multiple sessions or agents.
    """
    
    # Initialize WRITER model (Palmyra Medical)
    writer_model = WriterModel(
        client_args={"api_key": os.getenv("WRITER_API_KEY")}, model_id="palmyra-med"
    )
    med_agent = Agent(
        model=writer_model,
        system_prompt=MEDICAL_ASSISTANT_PROMPT,
        tools=[mem0_memory],
    )

    try:
        response = med_agent(f"Generate a medical plan or concept for: {topic}", user_id=user_id)
        output_text = str(response)
        
        # Mem0 automatically stores and indexes the result.
        print(f"Memory updated for user '{user_id}' at {datetime.now(UTC).isoformat()}")
        return output_text
    
    except Exception as e:
        return f"[Medical Assistant Error] {str(e)}"