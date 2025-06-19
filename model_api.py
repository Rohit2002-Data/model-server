from fastapi import FastAPI, Request
import requests
import os

# Hugging Face Inference API setup
HF_API_URL = "https://api-inference.huggingface.co/models/distilgpt2"
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "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")  # REPLACE this or use .env
HF_HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}

# FastAPI app instance
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hugging Face distilgpt2 API is ready."}

@app.post("/generate/")
async def generate(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt", "").strip()

        if not prompt:
            return {"error": "Prompt is empty"}

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 30,
                "do_sample": True,
                "temperature": 0.7
            }
        }

        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload)
        result = response.json()

        # Handle response
        if isinstance(result, list) and "generated_text" in result[0]:
            return {"response": result[0]["generated_text"]}
        else:
            return {"error": result}

    except Exception as e:
        return {"error": f"Exception occurred: {str(e)}"}
