"""
AI Chatbot - Chainlit Application
----------------------------------
Intelligent assistant that routes queries to:
  1. Direct LLM answer  (general knowledge / creative)
  2. RAG knowledge base (internal org docs — LLM signals intent via JSON)
  3. MCP platform tools (automation — LLM decides via tool call)

RAG routing: The LLM returns a structured JSON action block (not a function tool
call) to signal it wants a knowledge base search. This avoids a conflict where
LiteLLM's MCP auto-execution (require_approval: never) would attempt to run
search_knowledge_base as an MCP tool and fail.

If an MCP tool call has missing parameters, a DynamicForm JSX element
is rendered to collect them from the user before execution.
"""

import os
import json
import re
import httpx
import chainlit as cl
from dotenv import load_dotenv

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────
AI_MODEL                    = os.getenv("AI_MODEL", "gpt-oss-20b")
AI_EMBEDDING_MODEL          = os.getenv("AI_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
AI_GATEWAY_URL              = os.getenv("AI_GATEWAY_URL", "").rstrip("/")
AI_API_KEY                  = os.getenv("AI_API_KEY", "")
AI_GATEWAY_SSL_VERIFY       = os.getenv("AI_GATEWAY_SSL_VERIFY", "true").lower() != "false"
AI_MODEL_TEMPERATURE        = float(os.getenv("AI_MODEL_TEMPERATURE", "0.7"))
AI_MODEL_MAX_TOKENS         = int(os.getenv("AI_MODEL_MAX_TOKENS", "1024"))
AI_MODEL_CHAT_STREAMS       = os.getenv("AI_MODEL_CHAT_STREAMS", "false").lower() == "true"
AI_MODEL_RAG_TOP_K          = int(os.getenv("AI_MODEL_RAG_TOP_K", "5"))
AI_MODEL_RAG_SIMILARITY_THRESHOLD = float(os.getenv("AI_MODEL_RAG_SIMILARITY_THRESHOLD", "0.7"))

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an intelligent platform assistant that helps users with general knowledge,
organizational information, and infrastructure automation tasks.

You have access to the following capabilities:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CAPABILITY 1: KNOWLEDGE BASE SEARCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USE when the user asks about:
  - Company policies, HR, compliance, security standards
  - Internal naming conventions, architecture standards, runbooks
  - Anything prefixed with "our", "company", "internal", "org"

DO NOT USE when:
  - The query is general/factual knowledge (math, science, public tech docs)
  - The intent is to execute an action (create, provision, deploy)
  - The question is creative (writing, summarization)

TO TRIGGER A KNOWLEDGE BASE SEARCH, respond with ONLY this JSON
(no markdown fences, no surrounding text, just the raw JSON object):

{
  "action": "search_knowledge_base",
  "query": "<concise, optimised search query>"
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CAPABILITY 2: PLATFORM AUTOMATION  (MCP tools)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USE when the user intends to CREATE, PROVISION, DEPLOY, GENERATE, or AUTOMATE.
A matching tool must exist in your available tools list. Never hallucinate tools.

CRITICAL — MISSING PARAMETERS RULE:
If you identify an MCP tool to call but required parameters are NOT fully provided,
you MUST respond with ONLY the following JSON (no surrounding text, no markdown):

{
  "action": "request_form_input",
  "tool": "<exact_tool_name>",
  "server_name": "<server_name>",
  "title": "<human readable action title>",
  "description": "<brief description of what this action will do>",
  "fields": [
    {
      "id": "<parameter_name>",
      "label": "<Human Readable Label>",
      "type": "<text|number|select|textarea>",
      "required": true,
      "value": "<pre-filled if user provided it, else null>",
      "placeholder": "<helpful placeholder>",
      "options": ["opt1", "opt2"]
    }
  ]
}

PARAMETERS ARE SUFFICIENT: If all required params are provided in the user message,
call the tool immediately. Extract values intelligently from natural language.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CAPABILITY 3: DIRECT ANSWER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USE for: general knowledge, math, public tech concepts, creative tasks,
         or when no tool is applicable.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECISION FLOW — apply in this exact order for every message:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1: Is this general knowledge or a creative request?
  YES → Answer directly. Stop. No tools.

STEP 2: Is the intent to CREATE, PROVISION, DEPLOY, or AUTOMATE?
  YES → Find matching MCP tool
        All params present → Call the tool directly
        Params missing    → Return form request JSON
        No tool matches   → Go to STEP 3

STEP 3: Does this require internal/organizational knowledge?
  YES → Return the search_knowledge_base JSON signal (see CAPABILITY 1)
  NO  → Answer directly from training knowledge

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BEHAVIOUR RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. NEVER emit both a search_knowledge_base JSON and an MCP tool call in the same response.
2. NEVER guess or hallucinate parameter values for MCP tools.
3. When knowledge base context is injected into the conversation, always cite the source document.
4. If the knowledge base returns no relevant results, say so clearly then answer from training data.
5. For MCP actions with real-world side-effects (PRs, provisioning), state
   what you are about to do before calling the tool.
6. Only use MCP tools that are explicitly listed in your available tools.
"""

# NOTE: search_knowledge_base is NOT defined as a function tool.
# The LLM signals RAG intent via a JSON action block in its response content.
# This prevents LiteLLM (require_approval: never) from auto-executing it as MCP.


# ─── HTTP Helpers ─────────────────────────────────────────────────────────────
def _headers() -> dict:
    return {
        "Authorization": f"Bearer {AI_API_KEY}",
        "Content-Type": "application/json",
    }


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(verify=AI_GATEWAY_SSL_VERIFY, timeout=60.0)


# ─── Gateway API Calls ────────────────────────────────────────────────────────
async def fetch_mcp_servers() -> list:
    """Fetch available MCP servers from the gateway."""
    try:
        async with _client() as client:
            resp = await client.get(
                f"{AI_GATEWAY_URL}/v1/mcp/server",
                headers=_headers()
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        print(f"[WARN] Could not fetch MCP servers: {e}")
        return []


def build_mcp_tool_entries(mcp_servers: list) -> list:
    """
    Convert each MCP server into an entry in the tools array.
    Format required by the gateway for MCP tool calling.
    NOTE: Cannot coexist with rag.* in the same payload (NOTE-4b).
    """
    entries = []
    for server in mcp_servers:
        server_name = server.get("server_name", "")
        if not server_name:
            continue
        entries.append({
            "type": "mcp",
            "server_label": "litellm",
            "require_approval": "never",
            "server_url": f"litellm_prox/mcp/{server_name}"
        })
    return entries


async def call_llm(messages: list, tools: list = None) -> dict:
    """
    POST to /v1/chat/completions.
    tools must NOT include rag.* when tools array is present (NOTE-4b).
    """
    payload = {
        "model": AI_MODEL,
        "messages": messages,
        "temperature": AI_MODEL_TEMPERATURE,
        "max_tokens": AI_MODEL_MAX_TOKENS,
        "stream": AI_MODEL_CHAT_STREAMS,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    async with _client() as client:
        resp = await client.post(
            f"{AI_GATEWAY_URL}/v1/chat/completions",
            json=payload,
            headers=_headers()
        )
        resp.raise_for_status()
        return resp.json()


async def call_rag_query(query: str, include_sources: bool = True) -> dict:
    """
    POST to /v1/rag/query — full RAG pipeline handled by gateway.
    Called when the LLM requests search_knowledge_base.
    """
    payload = {
        "query": query,
        "top_k": AI_MODEL_RAG_TOP_K,
        "similarity_threshold": AI_MODEL_RAG_SIMILARITY_THRESHOLD,
        "include_sources": include_sources,
        "temperature": AI_MODEL_TEMPERATURE,
        "max_tokens": AI_MODEL_MAX_TOKENS,
        "user_api_key": AI_API_KEY,
    }
    async with _client() as client:
        resp = await client.post(
            f"{AI_GATEWAY_URL}/v1/rag/query",
            json=payload,
            headers=_headers()
        )
        resp.raise_for_status()
        return resp.json()


# ─── Helper Functions ─────────────────────────────────────────────────────────
# Known action values the LLM may signal via JSON response content
_KNOWN_ACTIONS = {"search_knowledge_base", "request_form_input"}


def try_parse_intent_signal(content: str) -> dict | None:
    """
    Check if the LLM's response is an intent-signal JSON.
    Handles two known action types:
      - "search_knowledge_base" : LLM wants a RAG search
      - "request_form_input"    : LLM needs MCP params from the user
    Returns the parsed dict if valid, None otherwise.
    """
    if not content or '"action"' not in content:
        return None
    try:
        # Extract the outermost JSON object from the content
        match = re.search(r'\{[^{}]*"action"\s*:[^{}]*\}', content, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            if parsed.get("action") in _KNOWN_ACTIONS:
                return parsed
    except (json.JSONDecodeError, AttributeError):
        pass
    return None


def format_rag_context(rag_data: dict) -> str:
    """Convert RAG query response into a context string for the LLM synthesis call."""
    sources = rag_data.get("sources", [])
    answer = rag_data.get("answer", "")

    if not sources and not answer:
        return "No relevant information found in the knowledge base."

    lines = ["Retrieved context from the knowledge base:", "---"]
    for i, src in enumerate(sources, 1):
        doc = src.get("metadata", {}).get("source", "Unknown")
        collection = src.get("collection", "")
        score = src.get("score", 0)
        text = src.get("text", "")
        lines.append(f"[{i}] {doc} (collection: {collection}, relevance: {score:.2f})")
        lines.append(text)
        lines.append("---")

    # Fallback: if no chunks but RAG service returned a direct answer
    if not sources and answer:
        lines.append(answer)

    return "\n".join(lines)


# ─── Tool Call Handlers ───────────────────────────────────────────────────────
async def handle_rag_signal(signal: dict, history: list, all_tools: list) -> str:
    """
    Handle a search_knowledge_base intent signal from the LLM.
    The LLM returned a JSON action block (not a function tool call), so there
    is no tool_call_id. RAG context is injected as a system message instead.

      1. Call /v1/rag/query with the LLM's search query
      2. Inject retrieved context into history as a system message
      3. Make a second /v1/chat/completions call to synthesize the final answer
    Returns the final answer string.
    """
    query = signal.get("query", "")

    async with cl.Step(name="Searching knowledge base", type="tool") as step:
        step.input = query
        rag_data = await call_rag_query(query=query, include_sources=True)
        retrieved = rag_data.get("retrieved_count", len(rag_data.get("sources", [])))
        step.output = f"Retrieved {retrieved} chunks"

    context = format_rag_context(rag_data)

    # Inject RAG context as a system message (no tool_call_id needed)
    history.append({
        "role": "system",
        "content": f"[Knowledge Base Context — use this to answer the user's question]\n{context}",
    })

    # Second LLM call — synthesize answer from injected context
    # Pass all_tools so MCP tools remain available for follow-up turns
    final_resp = await call_llm(messages=history, tools=all_tools)
    return final_resp["choices"][0]["message"]["content"]


async def handle_form_request(form_request: dict, history: list, all_tools: list) -> str:
    """
    Show DynamicForm.jsx to collect missing MCP parameters.
    After submission, re-call LLM with the collected values.
    Returns the final answer string.
    """
    form_element = cl.CustomElement(
        name="DynamicForm",
        props={
            "title": form_request.get("title", "Provide Details"),
            "description": form_request.get("description", ""),
            "tool": form_request.get("tool", ""),
            "server_name": form_request.get("server_name", ""),
            "fields": form_request.get("fields", []),
        }
    )

    form_response = await cl.AskElementMessage(
        content="I need a few more details to proceed:",
        element=form_element,
        timeout=300,
    ).send()

    if not form_response or form_response.get("cancelled"):
        return "Action cancelled."

    # Build collected values (strip internal keys)
    _skip_keys = {"submitted", "cancelled", "tool", "server_name"}
    collected = {k: v for k, v in form_response.items() if k not in _skip_keys}

    tool_name = form_request.get("tool", "the requested action")
    params_text = "\n".join(f"- {k}: {v}" for k, v in collected.items())
    follow_up = f"Proceed with {tool_name} using these parameters:\n{params_text}"

    # Append the form response as the user's next message, then call LLM
    history.append({"role": "user", "content": follow_up})

    async with cl.Step(name=f"Executing {tool_name}", type="tool") as step:
        step.input = json.dumps(collected, indent=2)
        final_resp = await call_llm(messages=history, tools=all_tools)
        answer = final_resp["choices"][0]["message"]["content"]
        step.output = answer

    return answer


# ─── Chainlit Lifecycle ───────────────────────────────────────────────────────
@cl.on_chat_start
async def on_chat_start():
    """
    Runs once per session.
    Fetches MCP servers, builds the combined tools list, initialises history.
    """
    # Fetch MCP servers (tool discovery — run once, cache in session)
    mcp_servers = await fetch_mcp_servers()
    mcp_tool_entries = build_mcp_tool_entries(mcp_servers)

    # MCP tools only — search_knowledge_base is handled via intent-signal JSON,
    # NOT as a function tool, to avoid LiteLLM auto-execution conflict.
    all_tools = mcp_tool_entries

    cl.user_session.set("message_history", [{"role": "system", "content": SYSTEM_PROMPT}])
    cl.user_session.set("all_tools", all_tools)
    cl.user_session.set("mcp_servers", mcp_servers)

    # Greeting with available capabilities
    server_labels = [s.get("alias") or s.get("server_name", "") for s in mcp_servers]
    tools_line = f"\n- ⚙️ Automation tools: {', '.join(server_labels)}" if server_labels else ""

    await cl.Message(
        content=(
            "👋 Hello! I'm your platform assistant.\n\n"
            "I can help you with:\n"
            "- 💬 General questions and technical assistance\n"
            "- 📚 Internal documentation and organizational policies\n"
            f"- 🚀 Platform automation tasks{tools_line}\n\n"
            "How can I help you today?"
        )
    ).send()


@cl.on_message
async def on_message(msg: cl.Message):
    """
    Main message handler.

    Flow:
      1. Append user message to history
      2. Call LLM with MCP tools only
      3. Route based on finish_reason + content:
         - stop + search_knowledge_base JSON → call RAG, second LLM call
         - stop + request_form_input JSON    → show DynamicForm, re-call LLM
         - stop + plain text                 → display answer directly
         - tool_calls                        → MCP auto-executed by LiteLLM
      4. Append assistant answer to history and persist
    """
    history: list = cl.user_session.get("message_history")
    all_tools: list = cl.user_session.get("all_tools")

    history.append({"role": "user", "content": msg.content})

    answer = None

    try:
        # ── First LLM call ────────────────────────────────────────────────────
        response = await call_llm(messages=history, tools=all_tools)
        choice = response["choices"][0]
        finish_reason = choice["finish_reason"]
        assistant_msg = choice["message"]

        # ── Route on finish_reason ────────────────────────────────────────────
        if finish_reason == "stop":
            content = assistant_msg.get("content", "")
            signal = try_parse_intent_signal(content)

            if signal and signal.get("action") == "search_knowledge_base":
                # ── RAG path ──────────────────────────────────────────────────
                # LLM returned a JSON signal requesting a knowledge base search.
                # Do NOT add the raw JSON to visible history — replace it with
                # the synthesized answer after RAG completes.
                answer = await handle_rag_signal(signal, history, all_tools)

            elif signal and signal.get("action") == "request_form_input":
                # ── MCP form path ─────────────────────────────────────────────
                # LLM signalled it needs missing MCP parameters from the user.
                history.append({"role": "assistant", "content": content})
                answer = await handle_form_request(signal, history, all_tools)

            else:
                # ── Direct answer ─────────────────────────────────────────────
                answer = content

        elif finish_reason == "tool_calls":
            # MCP tools with require_approval: "never" are auto-executed by LiteLLM.
            # finish_reason == "tool_calls" means LiteLLM did NOT auto-execute
            # (should not normally happen with never). Surface the result anyway.
            history.append(assistant_msg)
            tool_calls = assistant_msg.get("tool_calls", [])

            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                answer = (
                    f"⚙️ **{tool_name}** called with:\n"
                    f"```json\n{json.dumps(tool_args, indent=2)}\n```"
                )

        elif finish_reason == "length":
            answer = (
                (assistant_msg.get("content") or "") +
                "\n\n*⚠️ Response was truncated. Try asking a more specific question.*"
            )

        elif finish_reason == "content_filter":
            answer = "⚠️ This request was blocked by the content policy."

        else:
            answer = assistant_msg.get("content") or "I encountered an unexpected response. Please try again."

    except httpx.HTTPStatusError as e:
        answer = f"⚠️ Gateway error ({e.response.status_code}): {e.response.text[:200]}"
    except httpx.ConnectError:
        answer = "⚠️ Could not connect to the AI Gateway. Please check `AI_GATEWAY_URL` in your `.env`."
    except Exception as e:
        answer = f"⚠️ An unexpected error occurred: {str(e)}"

    # ── Send answer and persist history ──────────────────────────────────────
    if answer:
        await cl.Message(content=answer).send()
        history.append({"role": "assistant", "content": answer})

    cl.user_session.set("message_history", history)
