from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("distilgpt2").to("cpu")
model.eval()

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Model API is ready."}

@app.post("/generate/")
async def generate(request: Request):
    data = await request.json()
    input_ids = data.get("input_ids")
    attention_mask = data.get("attention_mask")

    if not input_ids:
        return {"error": "input_ids are missing"}

    inputs = {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask) if attention_mask else None
    }

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=0.7)

    return {"output_ids": output.tolist()}
