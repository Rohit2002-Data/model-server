from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM
import torch
import gc
import os

# Force no CUDA, no unnecessary warnings
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
torch.set_grad_enabled(False)

# Initialize FastAPI
app = FastAPI()

# Load distilgpt2 model in ultra-low RAM mode
model = AutoModelForCausalLM.from_pretrained(
    "distilgpt2",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
model.to("cpu")
model.eval()

@app.get("/")
def root():
    return {"message": "Ultra-Optimized distilgpt2 Model API is running."}

@app.post("/generate/")
async def generate(request: Request):
    try:
        data = await request.json()
        input_ids = data.get("input_ids")
        attention_mask = data.get("attention_mask")

        if not input_ids:
            return {"error": "Missing input_ids"}

        # Convert to torch tensors (single batch, no float ops)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        attn_tensor = torch.tensor([attention_mask], dtype=torch.long) if attention_mask else None

        # Generation (no gradients, small output)
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_tensor,
                attention_mask=attn_tensor if attn_tensor is not None else None,
                max_new_tokens=15,                   # 🔥 smaller generation
                do_sample=False,                     # 🔥 no sampling = less compute
                temperature=0.7,
                pad_token_id=model.config.eos_token_id  # to avoid pad_token errors
            )

        # Return raw output token IDs
        return {"output_ids": output.tolist()}

    except Exception as e:
        return {"error": f"Exception: {str(e)}"}

    finally:
        # 🔥 Always free memory explicitly
        del input_tensor, attn_tensor, output
        gc.collect()
