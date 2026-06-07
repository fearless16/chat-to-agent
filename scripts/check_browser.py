"""Minimal Firefox browser test."""
import sys
from playwright.sync_api import sync_playwright

print("Launching Firefox headless...")
p = sync_playwright().start()
try:
    b = p.firefox.launch(headless=True)
    print("OK: browser launched")
    page = b.new_page()
    page.goto("https://httpbin.org/ip", timeout=15000)
    print(f"OK: page loaded, content={page.content()[:200]}")
    b.close()
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
finally:
    p.stop()
