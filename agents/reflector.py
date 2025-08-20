def reflector(model, tokenizer, state: dict) -> dict:
    story = state.get("draft_story", "")
    if "incomplete" in story.lower():
        status = "needs_info"
    elif "weak" in story.lower():
        status = "refine_required"
    else:
        status = "satisfactory"

    reflection = f"[Mocked] Reflection completed. Status: {status}"

    return {
        **state,
        "reflection": reflection,
        "status": status
    }