"""
Deep Tutor Agent

Tutor agent using create_deep_agent with FilesystemBackend.

Personality and teaching guidelines are loaded from AGENTS.md via MemoryMiddleware.
Module skills are loaded from /skills/ via SkillsMiddleware (progressive disclosure).
The dynamic prompt injects per-session context: student name, module, and goals.

Assistant context (set per assistant in LangSmith):
- module_id:       which module to load (e.g. "module-1")
- store_namespace: student's Store namespace (derived from email, e.g. "jane@example_com")
"""

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.permissions import FilesystemPermission
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Routed through the LLM Gateway's custom-provider "alex-gpt-gateway" config,
# which pins every call to its configured upstream model regardless of the
# model name passed here (https://gateway.smith.langchain.com/models/{configName}).
model = init_chat_model(
    model="gpt-4.1",
    model_provider="openai",
    base_url="https://gateway.smith.langchain.com/models/alex-gpt-gateway",
    # this should be LANGSMITH_API_KEY (since we're calling the model through gateway)
    # but LANGSMITH_API_KEY is a reserved name and will be stripped by the CLI on deploy.
    # we use this instead - see .env.example
    api_key=os.environ["DEEP_TUTOR_API_KEY"],
)
# model = init_chat_model("openai:gpt-4o")
# model = init_chat_model("google_genai:gemini-2.5-flash")


# ---------------------------------------------------------------------------
# Context schema
# ---------------------------------------------------------------------------
@dataclass
class ContextSchema:
    module_id: str = "module-1"
    store_namespace: str = ""  # derived from student email


# ---------------------------------------------------------------------------
# Dynamic system prompt
# ---------------------------------------------------------------------------
@dynamic_prompt
async def tutor_prompt(request: ModelRequest) -> str:
    module_id = request.runtime.context.module_id
    store_namespace = request.runtime.context.store_namespace
    student_name = "Student"
    goals = ""

    if store_namespace and request.runtime.store:
        item = await request.runtime.store.aget((store_namespace,), key="profile")
        if item is not None:
            profile = item.value
            student_name = f"{profile.get('first_name', '')} {profile.get('last_name', '')}".strip() or "Student"
            goals = profile.get("goals", "")

    goals_line = f"\n\nThe student's goals: {goals}" if goals else ""
    return (
        f"Your student's name is {student_name}. "
        f"They are working on {module_id}.{goals_line}\n\n"
        f"On your first response, read /skills/{module_id}/SKILL.md. "
        f"That file contains your teaching instructions — follow them. "
        f"When you need to answer a factual question, read /skills/{module_id}/information.md."
    )


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------
_mcp_client = MultiServerMCPClient({
    "docs": {
        "url": "https://docs.langchain.com/mcp",
        "transport": "http",
    }
})
mcp_tools = asyncio.run(_mcp_client.get_tools())


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
backend = FilesystemBackend(
    root_dir=Path(__file__).parent.parent,
    virtual_mode=True,
)

graph = create_deep_agent(
    model=model,
    tools=[*mcp_tools],
    middleware=[tutor_prompt],
    backend=backend,
    memory=["/AGENTS.md"],
    skills=["/skills/"],
    permissions=[
        FilesystemPermission(operations=["write"], paths=["/skills/**"], mode="deny"),
    ],
    context_schema=ContextSchema,
)
