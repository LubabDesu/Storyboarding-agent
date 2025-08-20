from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import torch

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