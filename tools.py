import os
import base64
import asyncio
from playwright.async_api import async_playwright

# Module-level browser state
_playwright = None
_browser = None
_page = None


async def open_browser(headless=False):
    """Launch Chromium browser and create a page."""
    global _playwright, _browser, _page
    _playwright = await async_playwright().start()
    _browser = await _playwright.chromium.launch(headless=headless)
    _page = await _browser.new_page()
    await _page.set_viewport_size({"width": 1280, "height": 720})
    return "Browser opened successfully."


async def close_browser():
    """Close the browser and clean up."""
    global _playwright, _browser, _page
    if _browser:
        await _browser.close()
    if _playwright:
        await _playwright.stop()
    _browser = None
    _page = None
    _playwright = None
    return "Browser closed."


async def navigate(url):
    """Navigate to a URL."""
    await _page.goto(url, wait_until="domcontentloaded")
    return f"Navigated to {url}"


async def go_back():
    """Go back in browser history."""
    await _page.go_back()
    return "Went back."


async def go_forward():
    """Go forward in browser history."""
    await _page.go_forward()
    return "Went forward."


async def reload():
    """Reload the current page."""
    await _page.reload()
    return "Page reloaded."


async def wait(ms):
    """Wait for a specified number of milliseconds."""
    await asyncio.sleep(ms / 1000)
    return f"Waited {ms}ms."


async def screenshot():
    """Take a screenshot and return it as a base64-encoded PNG string."""
    img_bytes = await _page.screenshot()
    return base64.b64encode(img_bytes).decode("utf-8")


async def get_page_url():
    """Return the current page URL."""
    return _page.url


async def click(x, y):
    """Click at specific pixel coordinates."""
    await _page.mouse.click(x, y)
    return f"Clicked at ({x}, {y})."


async def double_click(x, y):
    """Double-click at specific pixel coordinates."""
    await _page.mouse.dblclick(x, y)
    return f"Double-clicked at ({x}, {y})."


async def hover(x, y):
    """Hover at specific pixel coordinates."""
    await _page.mouse.move(x, y)
    return f"Hovered at ({x}, {y})."


async def scroll(direction, amount):
    """Scroll the page up or down by a pixel amount."""
    delta = -amount if direction == "up" else amount
    await _page.mouse.wheel(0, delta)
    return f"Scrolled {direction} by {amount}px."


async def drag(from_x, from_y, to_x, to_y):
    """Drag from one point to another."""
    await _page.mouse.move(from_x, from_y)
    await _page.mouse.down()
    await _page.mouse.move(to_x, to_y)
    await _page.mouse.up()
    return f"Dragged from ({from_x}, {from_y}) to ({to_x}, {to_y})."


async def type_text(text):
    """Type text into the currently focused element."""
    await _page.keyboard.type(text, delay=50)
    return f"Typed text: {text}"


async def type_email():
    """Type the Duolingo email from .env char-by-char into the focused input."""
    email = os.environ.get("DUOLINGO_EMAIL", "")
    if not email:
        return "Error: DUOLINGO_EMAIL not set in .env"
    for char in email:
        await _page.keyboard.type(char, delay=30)
    return "Email typed successfully."


async def type_password():
    """Type the Duolingo password from .env char-by-char into the focused input."""
    password = os.environ.get("DUOLINGO_PASSWORD", "")
    if not password:
        return "Error: DUOLINGO_PASSWORD not set in .env"
    for char in password:
        await _page.keyboard.type(char, delay=30)
    return "Password typed successfully."


async def press_key(key):
    """Press a keyboard key (Enter, Tab, Escape, ArrowDown, etc.)."""
    await _page.keyboard.press(key)
    return f"Pressed key: {key}"


async def wait_for_navigation(timeout=10000):
    """Wait for page navigation to complete."""
    try:
        await _page.wait_for_load_state("domcontentloaded", timeout=timeout)
        return "Navigation completed."
    except Exception:
        return "Timeout waiting for navigation."


# Registry mapping tool names to functions
TOOL_REGISTRY = {
    "open_browser": open_browser,
    "close_browser": close_browser,
    "navigate": navigate,
    "go_back": go_back,
    "go_forward": go_forward,
    "reload": reload,
    "wait": wait,
    "screenshot": screenshot,
    "get_page_url": get_page_url,
    "click": click,
    "double_click": double_click,
    "hover": hover,
    "scroll": scroll,
    "drag": drag,
    "type_text": type_text,
    "type_email": type_email,
    "type_password": type_password,
    "press_key": press_key,
    "wait_for_navigation": wait_for_navigation,
}
