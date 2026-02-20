import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL = "Featherless-Chat-Models/Mistral-7B-Instruct-v0.2"
ADAPTER = "UMUGABEKAZI/mistral-medical-chatbot"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
print("Ready!")

def respond(question, max_tokens=200):
    if not question.strip():
        return "Please enter a question."
    prompt = f"Instruction: Answer the following medical question. Question: {question} Response:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=int(max_tokens), do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

with gr.Blocks(title="MedBot") as demo:
    gr.HTML("<div style='text-align:center;padding:20px'><h1>🩺 MedBot</h1><p>Medical assistant fine-tuned on Mistral-7B with LoRA</p></div>")
    gr.HTML("<div style='background:#fef3c7;border:1px solid #fcd34d;border-radius:8px;padding:10px;color:#92400e;margin-bottom:10px;'>⚠️ For educational purposes only. Not medical advice.</div>")
    with gr.Row():
        with gr.Column(scale=3):
            q = gr.Textbox(label="Medical Question", placeholder="e.g. What causes pneumonia?", lines=3)
            with gr.Row():
                btn = gr.Button("Ask MedBot", variant="primary")
                clr = gr.Button("Clear")
            a = gr.Textbox(label="Response", lines=8, interactive=False)
        with gr.Column(scale=1):
            slider = gr.Slider(64, 300, value=200, step=32, label="Max tokens")
            gr.Examples(examples=[["What are symptoms of diabetes?"],["What is aspirin used for?"],["What causes hypertension?"]], inputs=q)
    btn.click(respond, [q, slider], a)
    q.submit(respond, [q, slider], a)
    clr.click(lambda: ("",""), outputs=[q, a])
    gr.Markdown("---\n**Model**: Mistral-7B + LoRA | **Dataset**: medical_meadow_medical_flashcards | **Training**: 1 epoch, lr=5e-5")

demo.launch()
