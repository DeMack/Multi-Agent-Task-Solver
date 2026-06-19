def strip_fences(text: str) -> str:
    # Claude occasionally wraps JSON in ```json ... ``` even when told not to.
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        inner = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
        text = "\n".join(inner)
    return text.strip()
