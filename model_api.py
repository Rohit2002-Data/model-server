from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM
import torch

# Use float16 and disable gradients to save memory
torch.set_grad_enabled(False)

# Load model in eval mode, with low memory footprint
model = AutoModelForCausalLM.from_pretrained(
    "distilgpt2",
    torch_dtype=torch.float16  # Use half precision
).to("cpu")
model.eval()

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Model API is ready (low-memory mode)."}

@app.post("/generate/")
async def generate(request: Request):
    data = await request.json()
    input_ids = data.get("input_ids")
    attention_mask = data.get("attention_mask")

    if not input_ids:
        return {"error": "input_ids are missing"}

    # Move tensors to CPU and cast to float16
    inputs = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long) if attention_mask else None
    }

    try:
        output = model.generate(
            **inputs,
            max_new_tokens=20,      # reduced tokens
            do_sample=True,
            temperature=0.7
        )
        return {"output_ids": output.tolist()}
    except RuntimeError as e:
        return {"error": f"Model ran out of memory: {str(e)}"}
