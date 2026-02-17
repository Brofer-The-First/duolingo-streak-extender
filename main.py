import os
import json
import re
import time
import asyncio
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from tools import TOOL_REGISTRY

load_dotenv()

client = OpenAI()

MAX_SCREENSHOTS = 3

SYSTEM_PROMPT = """You are a browser automation agent. Your goal is to log into Duolingo and complete one lesson to extend the daily streak (if daily streak is already extended, complete a lesson anyway).

You have access to Playwright browser tools to control a Chromium browser.

To log in, you MUST use the `type_email` and `type_password` tools which will securely enter the credentials from the environment. Do NOT attempt to type credentials manually with type_text.

Workflow:
1. Open the browser
2. Navigate to Duolingo's website
3. Find and click the login button
4. Click the email input field, then call type_email to enter the email
5. Click the password input field, then call type_password to enter the password
6. Submit the login form
7. Navigate to a lesson and complete it by answering all questions
8. When the lesson is fully completed, make sure you're back on the Duolingo dashboard page, then respond with DONE

Take screenshots frequently to understand the current page state. Use them to decide what to do next."""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "open_browser",
        "description": "Launch Chromium browser and create a page.",
        "parameters": {
            "type": "object",
            "properties": {
                "headless": {"type": "boolean", "description": "Run in headless mode. Default false."}
            },
        },
    },
    {
        "type": "function",
        "name": "close_browser",
        "description": "Close the browser and clean up.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "navigate",
        "description": "Navigate to a URL.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "The URL to navigate to."}},
            "required": ["url"],
        },
    },
    {
        "type": "function",
        "name": "go_back",
        "description": "Go back in browser history.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "go_forward",
        "description": "Go forward in browser history.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "reload",
        "description": "Reload the current page.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "wait",
        "description": "Wait for a specified number of milliseconds.",
        "parameters": {
            "type": "object",
            "properties": {"ms": {"type": "integer", "description": "Milliseconds to wait."}},
            "required": ["ms"],
        },
    },
    {
        "type": "function",
        "name": "screenshot",
        "description": "Take a screenshot of the current page. Returns a base64-encoded PNG image. Use this to understand the current page state before deciding what to do.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "get_page_url",
        "description": "Return the current page URL.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "click",
        "description": "Click at specific pixel coordinates on the page. Use screenshot to determine the coordinates of the element you want to click.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "X coordinate."},
                "y": {"type": "integer", "description": "Y coordinate."},
            },
            "required": ["x", "y"],
        },
    },
    {
        "type": "function",
        "name": "double_click",
        "description": "Double-click at specific pixel coordinates.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "X coordinate."},
                "y": {"type": "integer", "description": "Y coordinate."},
            },
            "required": ["x", "y"],
        },
    },
    {
        "type": "function",
        "name": "hover",
        "description": "Hover at specific pixel coordinates.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "X coordinate."},
                "y": {"type": "integer", "description": "Y coordinate."},
            },
            "required": ["x", "y"],
        },
    },
    {
        "type": "function",
        "name": "scroll",
        "description": "Scroll the page up or down.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["up", "down"], "description": "Scroll direction."},
                "amount": {"type": "integer", "description": "Pixels to scroll."},
            },
            "required": ["direction", "amount"],
        },
    },
    {
        "type": "function",
        "name": "drag",
        "description": "Drag from one point to another.",
        "parameters": {
            "type": "object",
            "properties": {
                "from_x": {"type": "integer"}, "from_y": {"type": "integer"},
                "to_x": {"type": "integer"}, "to_y": {"type": "integer"},
            },
            "required": ["from_x", "from_y", "to_x", "to_y"],
        },
    },
    {
        "type": "function",
        "name": "type_text",
        "description": "Type text into the currently focused element.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Text to type."}},
            "required": ["text"],
        },
    },
    {
        "type": "function",
        "name": "type_email",
        "description": "Securely type the Duolingo email from environment variables char-by-char into the currently focused input field. Do NOT pass any arguments.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "type_password",
        "description": "Securely type the Duolingo password from environment variables char-by-char into the currently focused input field. Do NOT pass any arguments.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "press_key",
        "description": "Press a keyboard key (Enter, Tab, Escape, ArrowDown, ArrowUp, etc.).",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string", "description": "Key name."}},
            "required": ["key"],
        },
    },
    {
        "type": "function",
        "name": "wait_for_navigation",
        "description": "Wait for page navigation to complete.",
        "parameters": {
            "type": "object",
            "properties": {
                "timeout": {"type": "integer", "description": "Timeout in ms. Default 10000."},
            },
        },
    },
]


def is_screenshot_message(msg):
    """Check if a message is a screenshot (user message with input_image content)."""
    if not isinstance(msg, dict):
        return False
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if isinstance(content, list):
        return any(item.get("type") == "input_image" for item in content)
    return False


def trim_old_screenshots(messages):
    """Keep only the last MAX_SCREENSHOTS screenshot messages, replace older ones with a placeholder."""
    screenshot_indices = [i for i, msg in enumerate(messages) if is_screenshot_message(msg)]
    if len(screenshot_indices) <= MAX_SCREENSHOTS:
        return
    to_remove = screenshot_indices[:-MAX_SCREENSHOTS]
    for i in to_remove:
        messages[i] = {"role": "user", "content": "[old screenshot removed to save tokens]"}


async def run_tool(name, args):
    """Execute a tool by name with the given arguments."""
    func = TOOL_REGISTRY[name]
    return await func(**args)


async def main():
    messages: list[Any] = [{"role": "system", "content": SYSTEM_PROMPT}]
    iteration = 0

    print("Starting Duolingo Streak Extender agent...")
    print("Press Ctrl+C to stop.\n")

    while True:
        iteration += 1
        print(f"--- Iteration {iteration} ---")

        trim_old_screenshots(messages)

        while True:
            try:
                response = client.responses.create(
                    model="gpt-4.1",
                    input=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                )
                break
            except RateLimitError as e:
                retry_match = re.search(r"try again in (\d+(\.\d+)?)s", str(e))
                wait_time = float(retry_match.group(1)) if retry_match else 60.0
                print(f"  Rate limited. Waiting {wait_time:.1f}s for TPM reset...")
                time.sleep(wait_time)

        # Process all output items
        tool_calls = []
        for item in response.output:
            messages.append(item)

            if item.type == "function_call":
                tool_calls.append(item)
            elif item.type == "message":
                for content_block in item.content:
                    if hasattr(content_block, "text"):
                        print(f"Agent: {content_block.text}")
                        if "DONE" in content_block.text.upper():
                            print("\nAgent signaled completion. Exiting.")
                            return

        if not tool_calls:
            continue

        # Execute all tool calls
        for tc in tool_calls:
            func_name = tc.name
            func_args = json.loads(tc.arguments) if tc.arguments else {}
            print(f"  Tool: {func_name}({func_args})")

            try:
                result = await run_tool(func_name, func_args)
            except Exception as e:
                result = f"Error: {e}"

            # Handle screenshot results — send as image in a follow-up user message
            if func_name == "screenshot" and not str(result).startswith("Error"):
                messages.append({
                    "type": "function_call_output",
                    "call_id": tc.call_id,
                    "output": "Screenshot taken. See the image below.",
                })
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{result}",
                        }
                    ],
                })
            else:
                messages.append({
                    "type": "function_call_output",
                    "call_id": tc.call_id,
                    "output": str(result),
                })


if __name__ == "__main__":
    asyncio.run(main())
