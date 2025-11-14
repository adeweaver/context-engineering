#!/usr/bin/env python3
"""
📁 Knowledge Agent — Multi-Agent System (Writer + Mem0)

A Strands-based orchestration agent that coordinates specialized assistants for:
- Creative tasks (palmyra-creative)
- Financial analysis (palmyra-fin)
- Medical reasoning (palmyra-med)

All agents share persistent memory via Mem0, allowing context continuity across sessions.
"""

import os
from datetime import datetime, UTC
from dotenv import load_dotenv
from strands import Agent, tool
from strands.models.writer import WriterModel
from strands_tools import mem0_memory

# Import specialized sub-agents
from creative_assistant import creative_assistant
from fin_assistant import fin_assistant
from med_assistant import med_assistant

load_dotenv()

KNOWLEDGE_AGENT_PROMPT = """
You are KnowledgeAssistant, the orchestrator coordinating specialized domain agents.

Your role:
- Route user questions to the correct specialized agent based on their domain.
- Maintain long-term context using Mem0 memory.
- Synthesize results into a cohesive final summary.

Routing Logic:
- If the query involves biology, symptoms, treatment, or diagnosis → use MedAgent.
- If the query involves money, investments, ROI, valuation, or forecasting → use FinAgent.
- If the query involves writing, ideation, or creative generation → use CreativeAgent.
Always use Mem0 memory to recall relevant prior information for the same user.
When done, store your summary back into Mem0 memory for future sessions.
"""

@tool(
    name="KnowledgeOrchestrator",
    description="Routes user queries to the appropriate specialized agent and synthesizes results."
)
def knowledge_orchestrator(topic: str, user_id: str = "default_user") -> str:
    """Coordinates domain agents and manages shared persistent memory."""
    print(f"\n [KnowledgeAgent] Coordinating multi-agent reasoning for query: {topic}\n")

    # Initialize orchestrator model (Palmyra X5 on Bedrock)
    writer_model = WriterModel(
        client_args={"api_key": os.getenv("WRITER_API_KEY")},
        model_id="palmyra-x5",
        temperature=0.2,
    )

    # Initialize orchestrator Agent
    knowledge_agent = Agent(
        model=writer_model,
        system_prompt=KNOWLEDGE_AGENT_PROMPT,
        tools=[mem0_memory, creative_assistant, fin_assistant, med_assistant],
    )

    # Retrieve any prior context from Mem0 memory
    print("Retrieving prior context from Mem0 memory...")
    try:
        past_memories = knowledge_agent.tool.mem0_memory(action="retrieve", query=topic, user_id=user_id)
        if past_memories and past_memories.get("results"):
            print(f"Found {len(past_memories['results'])} relevant prior memories.")
            memory_context = "\n".join(f"- {m['memory']}" for m in past_memories["results"])
        else:
            memory_context = "(No relevant prior memory found.)"
    except Exception as e:
        error_msg = str(e)
        if "ExpiredTokenException" in error_msg or "expired" in error_msg.lower():
            print("AWS session token has expired. Please refresh your AWS credentials.")
            memory_context = "(Memory unavailable - AWS credentials expired.)"
        else:
            print(f"Error retrieving memory: {error_msg}")
            memory_context = f"(Memory unavailable: {error_msg})"

    # Let the orchestrator dynamically route the query
    orchestrator_prompt = f"""
    MEMORY CONTEXT:
    {memory_context}

    USER QUERY:
    {topic}

    TASK:
    Decide which specialized agent to use, invoke it, and return a unified, domain-specific response.
    """

    print("Running orchestrator model (Palmyra-X5)...")
    response = knowledge_agent(orchestrator_prompt, user_id=user_id)
    output_text = str(response)

    # Persist synthesized summary back into shared memory
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    try:
        knowledge_agent.tool.mem0_memory(
            action="store",
            content=f"Session summary for '{topic}' at {timestamp}:\n{output_text[:800]}",
            user_id=user_id,
            metadata={"agent": "KnowledgeAgent", "topic": topic},
        )
        print("Stored session summary in persistent Mem0 memory.")
    except Exception as e:
        error_msg = str(e)
        if "ExpiredTokenException" in error_msg or "expired" in error_msg.lower():
            print("Could not store session summary - AWS credentials expired.")
        else:
            print(f"Could not store session summary: {error_msg}")

    return output_text

if __name__ == "__main__":
    print("\n📁 Knowledge Agent - Multi-Agent System 📁\n")

    user_id = "Ashley_example_user"
    topic1 = "Using a 10% discount rate, estimate the present value of $5 million annual cash flows over 5 years."
    topic2 = "Using a 10% discount rate, estimate the PV of $5M annual cash flows over 5 years."

    print(f"Running multi-agent collaboration for topic: {topic1}")
    print(f"User ID: {user_id}\n")

    try:
        # First run — baseline memory creation
        print("[RUN 1] Initial collaboration...")
        response_1 = knowledge_orchestrator(topic=topic1, user_id=user_id)
        print("\n=== FINAL SYNTHESIS (RUN 1) ===\n")
        print(response_1)

        # Check what's stored
        print("\n[CHECK] Retrieving stored session memories...")
        # Create a simple agent to access mem0_memory tool
        check_agent = Agent(
            model=WriterModel(
                client_args={"api_key": os.getenv("WRITER_API_KEY")},
                model_id="palmyra-x5",
            ),
            tools=[mem0_memory],
        )
        stored = check_agent.tool.mem0_memory(action="retrieve", query=topic1, user_id=user_id)
        if stored and stored.get("results"):
            print(f"Found {len(stored['results'])} memories stored for '{user_id}':")
            for m in stored["results"][:3]:
                print(f"- {m['memory']}")
        else:
            print("No stored memories found yet — check Mem0 setup.")

        # Second run — should recall prior context
        print("\n[RUN 2] Re-running with similar query...")
        response_2 = knowledge_orchestrator(topic=topic2, user_id=user_id)
        print("\n=== FINAL SYNTHESIS (RUN 2) ===\n")
        print(response_2)

    except Exception as e:
        print(f"\n Error during multi-agent collaboration: {str(e)}")
