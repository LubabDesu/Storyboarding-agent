from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage
from langchain import hub
from langchain_core.runnables import Runnable
from typing import Dict
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_mistralai import ChatMistralAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
import os, httpx
from dotenv import load_dotenv
import cv2
import json
load_dotenv()
import math
import hashlib
import requests
from IPython.display import Markdown, display
import numpy as np
from PIL import Image
import decord
from decord import VideoReader, cpu
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import cv2
import subprocess
import yt_dlp
from googleapiclient.discovery import build
from dotenv import load_dotenv
from pathlib import Path

# from models import model, processor


# Load environment variables from .env file
load_dotenv()

# === Load Environment Vars ===
MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
LANGFUSE_PUBLIC_KEY = os.environ["LANGFUSE_PUBLIC_KEY"]
LANGFUSE_SECRET_KEY = os.environ["LANGFUSE_SECRET_KEY"]
LANGFUSE_HOST = os.environ["LANGFUSE_HOST"] 


# Define helper functions 

def collect_frames(dir_path: str) -> list[dict]:
    """
    Scan a directory of extracted frames and return a list of dicts
    each with an absolute image_path for.
    """
    p = Path(dir_path).expanduser().resolve()
    frames = [{"image_path": str(fp)} for fp in sorted(p.glob("*.jpg"))]
    return frames

def search_video_ydl(query,num_videos) :
    youtube = build("youtube", "v3", developerKey=GOOGLE_API_KEY)

    request = youtube.search().list(
        q=query,
        part="snippet",
        type="video",
        videoDuration="short",
        maxResults=num_videos
    )
    response = request.execute()
    return response

    

def download_video_ydl(video_id, out_dir="videos/"):
    url = f"https://www.youtube.com/watch?v={video_id}"
    opts = {
      "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
      "outtmpl": f"{out_dir}/{video_id}.%(ext)s",
      "nocheckcertificate":True,
      "retries": 3
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info).rsplit(".",1)[0] + ".mp4"




def inference(model, processor, video_path, prompt, max_new_tokens=1024, total_pixels=2048 * 28 * 28, min_pixels=16 * 28 * 28):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"video": video_path, "total_pixels": total_pixels, "min_pixels": min_pixels},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    print("video input:", video_inputs[0].shape)
    num_frames, _, resized_height, resized_width = video_inputs[0].shape
    print("num of video tokens:", int(num_frames / 2 * resized_height / 28 * resized_width / 28))
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]

# Helper Functions Definitions

def download_video(url, dest_path):
    response = requests.get(url, stream=True)
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8096):
            f.write(chunk)
    print(f"Video downloaded to {dest_path}")


def get_video_frames(video_path, num_frames=64, cache_dir='.cache'):
    os.makedirs(cache_dir, exist_ok=True)

    video_hash = hashlib.md5(video_path.encode('utf-8')).hexdigest()
    if video_path.startswith('http://') or video_path.startswith('https://'):
        video_file_path = os.path.join(cache_dir, f'{video_hash}.mp4')
        if not os.path.exists(video_file_path):
            download_video(video_path, video_file_path)
    else:
        video_file_path = video_path

    frames_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_frames.npy')
    timestamps_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_timestamps.npy')

    if os.path.exists(frames_cache_file) and os.path.exists(timestamps_cache_file):
        frames = np.load(frames_cache_file)
        timestamps = np.load(timestamps_cache_file)
        return video_file_path, frames, timestamps

    vr = VideoReader(video_file_path, ctx=cpu(0))
    total_frames = len(vr)

    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    timestamps = np.array([vr.get_frame_timestamp(idx) for idx in indices])

    np.save(frames_cache_file, frames)
    np.save(timestamps_cache_file, timestamps)
    
    return video_file_path, frames, timestamps


def create_image_grid(images, num_columns=8):
    pil_images = [Image.fromarray(image) for image in images]
    num_rows = math.ceil(len(images) / num_columns)

    img_width, img_height = pil_images[0].size
    grid_width = num_columns * img_width
    grid_height = num_rows * img_height
    grid_image = Image.new('RGB', (grid_width, grid_height))

    for idx, image in enumerate(pil_images):
        row_idx = idx // num_columns
        col_idx = idx % num_columns
        position = (col_idx * img_width, row_idx * img_height)
        grid_image.paste(image, position)

    return grid_image

def timestamp_to_frame_index(timestamp: str, fps: float) -> int:
    hh, mm, ss = map(int, timestamp.split(":"))
    return int((mm * 60 + ss) * fps)

def reencode_to_h264(src_path: str, cache_dir: str = ".cache") -> str:
    """
    Given any video file, re-encode it to H.264 in cache_dir and return the new path.
    If the cached H.264 exists, just return that.
    """
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.basename(src_path)
    name, _ = os.path.splitext(base)
    dst_path = os.path.join(cache_dir, f"{name}.h264.mp4")
    
    if not os.path.exists(dst_path):
        cmd = [
            "ffmpeg", "-y",    # overwrite if exists
            "-i", src_path,
            "-c:v", "libx264",
            "-crf", "25",
            "-preset", "fast",
            "-c:a", "copy",
            dst_path
        ]
        print(f"🔄 Re-encoding {src_path} → {dst_path}")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print("Video successfully re-encoded!")
    return dst_path

def extract_frames_from_timestamps(video_path, timestamps, output_dir="frames_out"):
    os.makedirs(output_dir, exist_ok=True)
    
    h264_path = reencode_to_h264(video_path)

    # 1) Open the re-encoded video
    cap = cv2.VideoCapture(h264_path)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Failed to open video file: {h264_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    for i, (timestamp, description) in enumerate(timestamps):
        print(f"Timestamp : {timestamp}")
        frame_idx = timestamp_to_frame_index(timestamp, fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = cap.read()
        # print(f"Ret is currently {ret}")
        if not ret:
            print(f"⚠️ Could not read frame at {timestamp}")
            continue

        filename = f"{i+1:02d}_{timestamp.replace(':', '-')}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"✅ Saved frame at {timestamp} → {filepath}")

    cap.release()

# ============= Define Models and API Wrapper =============
google_search = GoogleSerperAPIWrapper()

# model_path = "Qwen/Qwen2.5-VL-7B-Instruct"

# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     # attn_implementation="flash_attention_2",
#     device_map="auto"
# )
# processor = AutoProcessor.from_pretrained(model_path)


# Try loading quantized Qwen VL

model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
save_path = "./Qwen2.5-VL-7B-quantized"

# Define 4-bit or 8-bit quantization settings
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Set False for 8-bit
    bnb_8bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=False
)

# Load model and processor
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Save the quantized model
model.save_pretrained(save_path)
processor.save_pretrained(save_path)


# <-------------------- Define Tools ------------------------>


from langchain_core.tools import tool
@tool 
def search_and_download_yt_videos(query:str, num_videos:int) -> str :
    """
    Search YouTube for `query`, download the top result as an MP4,
    and return the local file path.
     Args:
        query (str): The search term for the videos.
        num_videos (int): The exact number of videos to search for and download. 
                         You must decide this value based on the user's request.
    """
    print(f"Agent decided to search for {num_videos} videos with query: '{query}'")
    response = search_video_ydl(query, num_videos) 
    items = response.get("items", [])
    selected = items[:num_videos]
    local_paths = []
    for item in selected : 
        video_id = item["id"]["videoId"]
        local_path = download_video_ydl(video_id)
        local_paths.append(local_path)
    return local_paths

@tool
def web_search(user_input_question: str) -> str:
    """Search Google for information."""
    return google_search.run(user_input_question)

@tool("analyze_and_extract", return_direct=True)
def analyze_and_extract(video_path: str) -> str:
    """Analyze a video and extract important frames or scenes."""
    
    #Extract duration from the video first 
    
    
    h264_path = reencode_to_h264(video_path)
#     print("👉 cwd        =", os.getcwd())
#     print("👉 target file=", h264_path)
#     print("👉 exists     =", os.path.exists(h264_path))
#     print("👉 is file    =", os.path.isfile(h264_path))
    cap = cv2.VideoCapture(h264_path)
   
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Total number of frames
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = num_frames / fps
    print(f"The duration of the video is {duration} seconds")
    
    cap.release()
    
    prompt     = ( f"""Please identify 5 important scenes that visually represent key moments, for image extraction for generating a storyboard.
    For each scene:
    
    The duration of the video is {duration} seconds long, so do not hallucinate any timestamps and ensure that all timestamps are within the video duration.

    Provide one timestamp only (in HH:MM:SS format)

    Ensure the timestamp is between 00:00 and 01:00

    If you’re unsure, choose the earliest moment that best visually captures the scene

    Include a short, clear description of what is happening at that timestamp

    Format the answer as 5 lines in the format: timestamp - description
    
    Start each timestamp - description pair on a new line, with no empty lines between any two timestamp - description pairs
       """
    )
    
    summary = inference(model, processor, video_path, prompt, total_pixels=1024 * 28 * 28,)
    print("Qwen-VL says:", summary)
    lines = [
        line.strip()
        for line in summary.splitlines()
        if line.strip() and "-" in line
    ]
    parsed = [tuple(line.split("-", 1)) for line in lines]
    # print(f"Parsed stuff is {parsed}\n\n")
    extract_frames_from_timestamps(video_path, parsed)
    frames = collect_frames("frames_out")
    items = [
    {
        "image_path": f["image_path"].strip(),
        "caption": cap.strip()
    }
    for f, (_, cap) in zip(frames, parsed)
    ]

    print(f"\nResult is {items}\n")

    return {"items": items}


# === Setup Model ===
custom_httpx_client = httpx.Client(
    verify=False,
    base_url='https://api.mistral.ai/v1',
    trust_env=True,
    headers={
        "Content-Type":  "application/json",
        "Accept":        "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
    },
    timeout=120,
)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0,
    client=custom_httpx_client,
    api_key=MISTRAL_API_KEY,
    streaming=True,
    callback_manager=callback_manager
)

# === Load Prompt and Bind Tools ===

tools = [web_search, analyze_and_extract, search_and_download_yt_videos]
agent_with_tools = llm.bind_tools(tools)
# prompt = hub.pull("hwchase17/openai-functions-agent")

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an intelligent agent with access to three tools:\n"
     "- `web_search`: Search the internet for up-to-date or factual information, like details on certain events.\n"
     "- `analyze_and_extract`: Analyze a video and extract out key frames to gather images for the storyboard\n"
     """- `search_and_download_yt_videos(query:str) -> str` : Searches Youtube for videos and downloads them, returning the file path of the downloaded videoes\n\n"""
     """If the user query mentions a video file or a .mp4 link, use the `analyze_video` tool, then use the "extract_frames" tool on the output to extract images for the storyboard generation.\n"""
     "If the user asks a research question or general query, or if the prompt does not contain much information, use the `web_search` tool.\n"
     "Be deliberate and avoid hallucinating — only invoke a tool if it's clearly needed. Also you do not have to follow the needs exactly word for word, reason and think what are the best keywords to search for based on the needs\n"
     """When you decide to use a tool, you MUST format exactly:

Action: one of {{tool_names}}
Action Input: a single valid JSON object

- Do NOT put JSON on the Action line.
- Do NOT wrap the tool name in brackets or quotes.
- Do NOT include bullets or prefixes like "[]".
- The Action line must be the tool name alone.
     """
     "If the input does not contain enough visual information like photos or videos, or you believe you need more videos, you must invoke the search_and_download_yt_videos tool, then analyze it.\n"
     "When needs are provided, you MUST build queries that explicitly include those phrases (or their synonyms)"
     """
     Based on the state of the current workflow, invoke the relevant tools : 
     """
     "STATE (from orchestrator):\n"
     "- reason_for_retry: {reason_for_retry}\n"
     "- needs: {needs}\n"
     "- attempt: {attempt}\n"
     "- search_history: {search_history}\n\n"

     """Your response MUST BE ONLY the raw, compiled JSON output from the tool. Do not add any conversational text, introductions, or summaries.

For example, if a tool returns a list of video frames, your entire output should look like this:


[
  {{"image_path": "/path/to/image1.jpg", "caption": "A description of the first frame."}},
  {{"image_path": "/path/to/image2.jpg", "caption": "A description of the second frame."}}
  ...
]

Do not include the word json as well, I want the output in the format of a list.
     """
     """
     SEARCH EXPANSION POLICY
1) Query construction
   - If needs is provided: build one query per need: "<topic> <shot> <user_query> Singapore NDP OR SG60".
   - If needs is empty: extract 2–3 keyphrases from reason_for_retry and build queries: "<keyphrase> <user_query> Singapore".
   - Always add variants across attempts:
     attempt==0: literal terms (exact topic/shot) + site:youtube.com when looking for videos.
     attempt==1: add synonyms/aliases and country keywords (Singapore|SG|NDP|SG60); increase result breadth.
     attempt>=2: broaden time/place, drop restrictive terms, add filetype filters, try alternative sources (news outlets, agencies).

2) De-duplication & history
   - Before executing, drop any query present in search_history.
   - After executing a query, append it to search_history.

3) Tool selection
   - If target is video or you need visual context, call search_and_download_yt_videos(query) for top matches (1–3), then for each downloaded path call analyze_and_extract(video_path).
   - If reason_for_retry indicates missing facts/context only (not visuals), call web_search first to refine terms, then re-issue YouTube queries.

4) Post-processing
   - Concatenate all analyze_and_extract.items results.
   - Normalize paths: remove stray spaces before extension (" .jpg" → ".jpg").
   - Remove duplicates by (image_path, caption).
   - If zero items found and attempt==0, broaden and retry once within this turn.
   - If still zero items, return an empty list (caller will loop).
     """

    ),
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder(variable_name="messages"),  # allows passing in user messages dynamically
    MessagesPlaceholder(variable_name="agent_scratchpad")
    
])

agent = create_tool_calling_agent(agent_with_tools, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, output_key="output" )


# === Helper Function for information_gatherer node === 

def parse_knowledge(output) -> list:
    """
    Robustly parses tool output that could be a JSON string,
    a dictionary containing a list, or a list itself.
    Always returns a list of items.
    """
   
    # Case 1: The output is a JSON string (most likely from the multi-video case)
    if isinstance(output, str):
        try:
            data = json.loads(output)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                # Handles cases where the string was '{"items": [...] }'
                return data.get("items", [])
            return [] # The JSON was valid but not a list or dict
        except json.JSONDecodeError:
            print("⚠️ Warning: Could not decode JSON string. Returning empty list.")
            return []

    # Case 2: The output is already a dictionary (likely from the single-video case)
    elif isinstance(output, dict):
        return output.get("items", [])

    # Case 3: The output is already a list
    elif isinstance(output, list):
        return output

    # Default case for any other unexpected type
    return []
# # === Info Gatherer Node ===

def information_gatherer(state: dict) -> dict:
    """
    Gathers information using the web_search agent and returns the findings.
    """
    print(f"📥 [information_gatherer] received state: {state}")


    query = state["input"]
    prev_know = state.get("knowledge", [])
    if not prev_know : 
        prev_know = []

    needs = state.get("needs", [])
    reason_for_retry = state.get("reason_for_retry", "")
    attempt = state.get("attempt", 0)
    search_history = state.get("search_history", [])

    # Now, pass the dynamic values into the executor
    result = executor.invoke({
        "input": query,
        "messages": [],
        "needs": needs,                # <-- Pass the dynamic needs here
        "attempt": attempt,            # <-- Pass the current attempt
        "search_history": search_history, # <-- Pass the search history
        "reason_for_retry": reason_for_retry # <-- Pass the reason
    })
    
    items = parse_knowledge(result['output'])
    combined = prev_know + items
    state["knowledge"] = combined

    print("🧠 [info_gatherer] accumulated knowledge:\n", combined)


    return state
