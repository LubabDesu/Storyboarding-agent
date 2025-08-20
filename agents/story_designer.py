def story_designer(state: dict) -> dict:
    info = state.get("info", "")
    draft_story = f"[Mocked] Created story scenes from info: {info}"

    return {
        **state,
        "draft_story": draft_story,
        "status": "reflect_needed"
    }