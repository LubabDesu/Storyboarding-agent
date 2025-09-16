import torch
import json
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import outlines

class StoryboardScene(BaseModel):
    """Represents a single scene or frame within the storyboard."""
    scene_index: int = Field(
        ..., 
        description="Sequential index of the scene, starting from 1.", 
        gt=0
    )
    image_path: str = Field(
        ..., 
        description="The exact file path of the image used for this scene."
    )
    frame_caption: str = Field(
        ..., 
        description="A concise, present-tense caption for the frame (max 20 words).", 
        max_length=150
    )
    shot_type: Literal[
        "close_up", "medium", "wide", "detail", "establishing", "over_the_shoulder"
    ] = Field(
        ..., 
        description="The type of camera shot."
    )
    reason: str = Field(
        ..., 
        description="A brief reason why this image was chosen for this narrative beat (max 15 words).", 
        max_length=100
    )

class FinalStoryboard(BaseModel):
    """The final, structured output for a complete storyboard."""
    narrative: str = Field(
        ..., 
        description="A 6-8 sentence narrative that sets the story's context and flow.", 
        min_length=50, 
        max_length=1200
    )
    storyboard: List[StoryboardScene] = Field(
        ..., 
        description="A list of scenes that make up the storyboard.", 
        min_items=1
    )
    notes: Optional[str] = Field(
        None, 
        description="Optional notes on risks, gaps, or suggestions for better asset coverage."
    )
    
    
JUDGE_PROMPT = """SYSTEM: Storyboard Planner

## Purpose
You generate a clear, visually-driven storyboard that uses ONLY the images provided.

## Inputs (provided by the caller)
- user_query: brief description of the goal/theme (e.g., “highlight final moments of the World Pool Championship”).
- items: a list of dictionaries: [{{"image_path": <abs/local path>, "caption": <short description>}}]. Use only these images.

## Rules: Best Practices for Effective Storyboarding
1) Layout efficiency & cleanliness
   - Use space intentionally; keep composition balanced and uncluttered.
2) Character & prop positioning
   - Favor expressive poses/gestures; ensure body language conveys the scene’s intent.
3) Cropping, sizing & layering
   - Imply depth (foreground/mid/background). Vary shot types (close/medium/wide) to maintain interest.
4) Color & effects
   - Use color/effects to indicate mood/time/flashback sparingly and consistently.
5) Consistency
   - Keep visual style, character colors, fonts, and recurring objects consistent across frames.
6) Campaign/Social pacing (when relevant)
   - Be concise; ensure each frame advances the message; keep captions short and scannable.
7) Review & refine
   - Ensure continuity, clarity, and alignment with the user_query; avoid unnecessary detail.

## Task
Given `user_query` and `items`, do the following:
A) Narrative: Write a concise storyline (4–8 sentences) that matches the user_query and could be illustrated by the provided images.
B) Image selection: Choose concrete images from `items` that best support the narrative beats. Do NOT invent URLs or assets. Use only the given `image_path` values.

## Constraints
- Do NOT reference images that are not in `items`.
- Do NOT rewrite or move images; just select and order them.
- Keep captions concise (≤ 20 words), active voice, present tense.
- If `items` are insufficient for a coherent flow, say so in `notes` and list the missing types of shots (e.g., “need a crowd wide shot”).

## Output format (JSON only; no extra text, I emphasize NO EXTRA TEXT)\n
{{
  "narrative": "<6–8 sentence narrative, no markdown>",
  "storyboard": [
    {{
      "scene_index": <int starting at 1>,
      "image_path": "<exact path from items>",
      "frame_caption": "<1–2 short sentences that align with the narrative>",
      "shot_type": "<close_up|medium|wide|detail|establishing|over_the_shoulder>",
      "reason": "<≤ 15 words on why this image serves this beat>"
    }}
    // ... repeat for 4–10 scenes
  ],
  "notes": "<optional: risks, gaps, or suggestions for better coverage>"
}}

## Selection guidance
- Prefer diversity: start with an establishing/wide, mix in medium/close-ups for key beats.
- Map each narrative sentence to one selected image; merge or split beats if images suggest better pacing.
- Prioritize images whose captions strongly match verbs/actions in the narrative.
- If multiple images are similar, pick the clearest one and avoid redundancy.

## Quality checks before returning
- Every storyboard.image_path exists in `items`.
- scene_index is strictly increasing from 1, no gaps.
- Captions are ≤ 20 words, present tense, no hashtags/emoji.
- Narrative and storyboard are consistent (same sequence of events).

Here is the user query : {query}
and Here is the items : {items}
"""


def qwen_prompt(user_query: str) -> str:
    return (
        f"You are a visual reasoning assistant. The user’s query is:\n"
        f"“{user_query}”.\n"
        "Describe what the image depicts in 1–2 sentences and briefly state "
        "why this scene is relevant to the query."
    )

def describe_with_qwen(qwen, frames, user_query) :
    prompt = qwen_prompt(user_query)
    out = []
    for f in frames:
        resp = qwen.describe(image=f["image_path"], prompt=prompt, spatial_json=False)
        cap = resp.get("caption", "").strip()
        out.append({**f, "qwen_caption": cap})
    return out

def build_candidates_with_qwen(state: dict) -> dict:
    """
    Expects in state:
      - state['user_query']: str
      - state['frames']: list[{'image_path': str, ...}]
      - state['qwen']: a Qwen 2.5 VL inference instance with .describe(...)
      - optional state['caption_cache']: dict[str, str]

    Produces:
      - state['candidates']: list[{'image_path': str, 'caption': str}]
    """
    user_query = state['user_query']
    frames = state["frames"]
    
    for f in frames : 
        try : 
            resp = qwen.describe(image=img_path, prompt=prompt)          
            print(f"for the image {f}, qwen said {resp}")
            caption = (resp.get("caption") or "").strip()
        except Exception as e:
            caption = "Image could not be captioned."
            
    candidates.append({"image_path": img_path, "caption": caption})

    state["caption_cache"] = cache
    state["candidates"] = candidates
    return state

def final_formatter(model, tokenizer, state: dict) -> dict:
    """
    Generates a structured storyboard using 'outlines' to guarantee valid output.
    """
    print("\n✅ Entering Final Formatter node with `outlines`......")

    try:
        # 1. Create the structured generation model from base model
        outlines_model = outlines.from_transformers(model, tokenizer)
        user_query = state["input"]
        items_str = json.dumps(state["knowledge"], indent=2)
        prompt = JUDGE_PROMPT.format(query=user_query, items=items_str)

        print("🧠 Generating structured response from language model...")

        # 2. Call the outlines generator with Pydantic schema
        # This directly returns a validated Pydantic object, not a string!
        json_text = outlines_model(prompt, output_type=FinalStoryboard, temperature=0.7, max_new_tokens=1024)

        print("✨ Successfully created and validated storyboard object!\n")
        print(json_text)
        # 3. Convert the Pydantic object to a dictionary for the final state
        

    except Exception as e:
        # This will now only catch major errors, not simple JSON parsing issues
        print(f"❌ An unexpected error occurred: {e}")
        final_output_dict = {"error": f"An unexpected error occurred: {str(e)}"}
        status = "error"

    print("----------------END OF FORMATTER------------------\n")

    return {
        **state,
        "final_output": json_text,
        "status": "satisfactory"
    }
