import gradio as gr
import os
import base64
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL"),
)

# Base64 encode image
def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except:
        return ""

# Header HTML
def get_header_html():
    if os.path.exists("static/aivn_logo.png"):
        logo_base64 = image_to_base64("static/aivn_logo.png")
        logo_html = f"""
            <img src="data:image/png;base64,{logo_base64}" 
                 alt="Logo" 
                 style="height: 120px; width: auto; margin-right: 20px; margin-bottom: 20px;">
        """
    else:
        logo_html = "<h2>AI VIETNAM</h2>"

    return f"""
    <div style="display: flex; align-items: center; padding: 20px;">
        <div>{logo_html}</div>
        <div style="margin-left: 20px;">
            <h1>🧠 LLM Chatbot</h1>
            <p>Conversational Assistant</p>
            <p style="color: #555;">🚀 AIO2024 Module 10 LLM Chatbot with RLHF</p>
        </div>
    </div>
    """

# Footer HTML
def get_footer_html():
    return """
    <div style="
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: white;
        padding: 10px;
        text-align: center;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        z-index: 1000;
        font-size: 14px;
    ">
        Created by <a href="https://vlai.work" target="_blank" style="color: #007BFF;">VLAI</a> • AI VIETNAM
    </div>
    """

# Chat function
def respond(message, history, system_message, max_tokens, temperature, top_p):
    messages = [{"role": "system", "content": system_message}]
    for msg in history:
        messages.append(msg)
    messages.append({"role": "user", "content": message})

    stream = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL"),
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    )

    response = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            response += token
            yield {"role": "assistant", "content": response}


# UI
chat = gr.ChatInterface(
    fn=respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly chatbot.", label="System message"),
        gr.Slider(1, 2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(0.1, 4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)")
    ],
    type="messages",
)


# Combine with layout elements
with gr.Blocks(css=".gradio-container {min-height: 100vh;}") as demo:
    gr.HTML(get_header_html())
    chat.render()
    gr.HTML(get_footer_html())

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    demo.launch(server_name="0.0.0.0", server_port=7860)
