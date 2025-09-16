# 🎬 Agentic Storyboarding with LLMs

An AI workflow that automatically generates scene-by-scene storyboards from raw video or textual prompts. Built with LangGraph, this project demonstrates how agentic architectures can combine reasoning, multimodal analysis, and reflection-inspired prompting to produce coherent narrative outputs.


## ✨ Overview

Storyboarding is central to video production, but it’s often time-consuming and manual. This project explores how multi-agent workflows powered by large language and vision models can:
	•	📸 Extract key frames and information from raw video
	•	📝 Design coherent storylines across multiple scenes
	•	🔄 Refine for consistency and coverage with reflection-inspired prompting
	•	📂 Output a structured storyboard object

This system showcases how agentic frameworks can orchestrate specialized models to perform complex, multi-step creative tasks.


## ⚙️ Architecture

<img width="453" height="416" alt="Pipeline Diagram" src="https://github.com/user-attachments/assets/7ffb405c-da1b-48a6-b0c7-e5b2935c8b40" />  


The pipeline is implemented in LangGraph and consists of:\
	•	Orchestrator – Routes tasks and manages overall flow (ready_to_design vs. needs_info).\
	•	Information Gatherer – Equipped with tools like web search, YouTube video scene extraction, and frame analysis (Qwen-VL 2.5).\
	•	Final Formatter – Drafts narrative scene descriptions, ensures logical flow, and outputs structured storyboard text for downstream use.

Models used:
	•	Qwen-VL 2.5 → Visual analysis.\
	•	Mistral → Orchestration & drafting.\
	•	Llama-3.1 → Reasoning & reflection within prompts.

📊 Key Features\
	•	🧩 Agentic workflow orchestration with LangGraph\
	•	🌐 Tool-equipped agents (web search + YouTube extraction)\
	•	🎥 Multimodal support: text + video frames\
	•	🔄 Reflection-inspired prompting to improve coherence


## 🛠️ Sample Workflow  

<img width="723" height="128" alt="orch_needs_info" src="https://github.com/user-attachments/assets/02c2bf67-7cd1-4698-94f7-a5c65eb12cb6" />  

**1. Orchestrator detects missing context**  
The orchestrator determines that the input (video/text) needs additional information before storyboarding can proceed.  

<br>  

<img width="707" height="321" alt="youtube_video_tool_call" src="https://github.com/user-attachments/assets/062e0708-9324-48d9-a3f0-dfef8ae56554" />  

**2. Information Gatherer calls YouTube tool**  
The system fetches video metadata and content using a YouTube extraction tool to prepare for frame-level analysis.  

<br>  

<img width="722" height="331" alt="analyze_extract_frame_toolcall" src="https://github.com/user-attachments/assets/deb887c1-1670-4ef9-a54a-e16611e048ec" />  

**3. Frame analysis: extracting candidate scenes (step 1)**  
Qwen-VL 2.5 processes raw frames to propose meaningful scene boundaries and captions.  

<br>  

<img width="707" height="220" alt="analyze_extract_frame_toolcall2" src="https://github.com/user-attachments/assets/ff9e68e1-586e-460d-9558-e76f81ba246e" />  

**4. Frame analysis: refining and validating selections (step 2)**  
The agent double-checks timestamps and scene content to ensure relevant frames are passed forward.  

<br>  

<img width="721" height="196" alt="orch_ready_to_design" src="https://github.com/user-attachments/assets/5ac6010b-0737-428f-b6b5-869dbddbca82" />  

**5. Orchestrator transitions to Final Formatter**  
Once enough context is gathered, the orchestrator switches to `ready_to_design`, handing off to the Final Formatter, which creates narrative scene descriptions and produces the polished storyboard output.  oduces the polished storyboard output.



## 📑 Sample Output  

✨ Successfully created and validated storyboard object!  

{\
  "narrative": "The 60th National Day Parade in Singapore kicked off with a grand military parade featuring tanks and soldiers marching in formation. As the parade progressed, a veteran saluted in front of a crowd, paying tribute to the nation's history. Next, a group of uniformed women stood together, likely part of a ceremonial unit. The formal event continued with a man in a military uniform saluting alongside others. The parade also highlighted community involvement, with firefighters waving to the crowd from a vehicle. Finally, the event concluded with a grand finale that brought the crowd together in celebration.",\
  "storyboard": [\
    {\
      "scene_index": 1,\
      "image_path": "/home/jovyan/frames_out/01_00-00-00.jpg",\
      "frame_caption": "Military parade kicks off with tanks and soldiers marching in formation.",\
      "shot_type": "wide",\
      "reason": "Establishing shot for the parade"\
    },\
    {\
      "scene_index": 2,\
      "image_path": "/home/jovyan/frames_out/02_00-00-10.jpg",\
      "frame_caption": "Veteran salutes in front of the crowd.",\
      "shot_type": "medium",\
      "reason": "Emphasize the veteran's respect for the nation"\
    },\
    {\
      "scene_index": 3,\
      "image_path": "/home/jovyan/frames_out/03_00-00-30.jpg",\
      "frame_caption": "Ceremonial unit stands together.",\
      "shot_type": "medium",\
      "reason": "Highlight the uniformed women's presence"\
    },\
    {\
      "scene_index": 4,\
      "image_path": "/home/jovyan/frames_out/04_00-00-40.jpg",\
      "frame_caption": "Formal salute by military personnel.",\
      "shot_type": "close_up",\
      "reason": "Emphasize the respect and unity"\
    },\
    {\
      "scene_index": 5,\
      "image_path": "/home/jovyan/frames_out/05_00-00-50.jpg",\
      "frame_caption": "Firefighters wave to the crowd from a vehicle.",\
      "shot_type": "medium",\
      "reason": "Showcase community involvement"\
    }\
  ],\
  "notes": "This is a test storyboard for the 60th National Day Parade in Singapore. The provided images cover the parade's military aspect, veteran tribute, ceremonial unit, formal salute, and community involvement. However, the storyboard could be improved by adding images that capture the grand finale and crowd celebration."\
}



## 🎨 Sample Storyboard Images

<img width="379" height="364" alt="storyboard_1" src="https://github.com/user-attachments/assets/e594dd4e-7739-4e3f-8593-9da45cb349c5" />  
<img width="311" height="366" alt="storyboard_2" src="https://github.com/user-attachments/assets/428626d8-3705-4a71-9026-902704247a7f" />  
<img width="338" height="390" alt="storyboard_3" src="https://github.com/user-attachments/assets/a3b4d76b-aac1-43a7-8ce9-f120db61e4b1" />  



