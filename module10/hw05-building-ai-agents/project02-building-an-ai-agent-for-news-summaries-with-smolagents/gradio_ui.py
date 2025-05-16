#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import shutil
import base64
from typing import Optional

import gradio as gr

from smolagents.agent_types import AgentAudio, AgentImage, AgentText
from smolagents.agents import MultiStepAgent, PlanningStep
from smolagents.memory import ActionStep, FinalAnswerStep, MemoryStep
from smolagents.utils import _is_package_available

CUSTOM_CSS = """
    .gradio-container {min-height: 100vh;} 
    .content-wrap {padding-bottom: 60px;}
    .full-width-btn {
        width: 100% !important;
        height: 50px !important;
        font-size: 18px !important;
        margin-top: 20px !important;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4) !important;
        color: white !important;
        border: none !important;
    }
    .full-width-btn:hover {
        background: linear-gradient(45deg, #FF5252, #3CB4AC) !important;
    }
    """


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def create_header():
    with gr.Row():
        with gr.Column(scale=1):
            if os.path.exists("static/aivn_logo.png"):
                logo_base64 = image_to_base64("static/aivn_logo.png")
                gr.HTML(f"""
                <img src="data:image/png;base64,{logo_base64}" 
                    alt="Logo" 
                    style="height: 120px; width: auto; margin-right: 20px; margin-bottom: 20px;">
                """)
            else:
                gr.HTML("""
                <div style="height: 120px; display: flex; align-items: center; justify-content: center; font-size: 24px; font-weight: bold;">
                    AI VIETNAM
                </div>
                """)
        with gr.Column(scale=4):
            gr.Markdown(
                """
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0 15px;">
                    <div>
                        <h1 style="margin-bottom:0;">📰 News Summary Agent</h1>
                        <p style="margin-top: 0.5em; color: #666;">🚀 AIO2024 Module 10 🤗</p>
                        <p style="margin-top: 0.5em; color: #2c3e50;">🗞️ Real-time News Fetch & Summarization</p>
                        <p style="margin-top: 0.2em; color: #7f8c8d;">🔍 Topic Classification & Insight Extraction</p>
                    </div>
                </div>
                """)


def create_footer():
    footer_html = """
    <style>
        .sticky-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background: white;
            padding: 10px;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
            z-index: 1000;
        }
        .content-wrap {
            padding-bottom: 60px; /* Footer height + extra spacing */
        }
    </style>
    
    <div class="sticky-footer">
        <div style="text-align: center; font-size: 14px;">
            Created by <a href="https://vlai.work" target="_blank" style="color: #007BFF; text-decoration: none;">VLAI</a>
            • AI VIETNAM
        </div>
    </div>
    """
    return gr.HTML(footer_html)


def get_step_footnote_content(step_log: MemoryStep, step_name: str) -> str:
    """Get a footnote string for a step log with duration and token information"""
    step_footnote = f"**{step_name}**"
    if hasattr(step_log, "input_token_count") and hasattr(step_log, "output_token_count"):
        token_str = f" | Input tokens:{step_log.input_token_count:,} | Output tokens: {step_log.output_token_count:,}"
        step_footnote += token_str
    if hasattr(step_log, "duration"):
        step_duration = f" | Duration: {round(float(step_log.duration), 2)}" if step_log.duration else None
        step_footnote += step_duration
    step_footnote_content = f""" < span style = "color: #bbbbc2; font-size: 12px;" > {step_footnote} < /span > """
    return step_footnote_content


def pull_messages_from_step(step_log: MemoryStep):
    """Extract ChatMessage objects from agent steps with proper nesting"""
    if not _is_package_available("gradio"):
        raise ModuleNotFoundError(
            "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
        )

    if isinstance(step_log, ActionStep):
        step_number = f"Step {step_log.step_number}" if step_log.step_number is not None else "Step"
        yield gr.ChatMessage(role="assistant", content=f"**{step_number}**")

        if hasattr(step_log, "model_output") and step_log.model_output:
            model_output = step_log.model_output.strip()
            model_output = re.sub(r"```\s*<end_code>", "```", model_output)
            model_output = re.sub(r"<end_code>\s*```", "```", model_output)
            model_output = re.sub(
                r"```\s*\n\s*<end_code>", "```", model_output)
            model_output = model_output.strip()
            yield gr.ChatMessage(role="assistant", content=model_output)

        if hasattr(step_log, "tool_calls") and step_log.tool_calls:
            first_tool_call = step_log.tool_calls[0]
            used_code = first_tool_call.name == "python_interpreter"
            args = first_tool_call.arguments
            content = str(args.get("answer", args)) if isinstance(
                args, dict) else str(args).strip()

            if used_code:
                content = re.sub(r"```.*?\n", "", content)
                content = re.sub(r"\s*<end_code>\s*", "", content).strip()
                if not content.startswith("```python"):
                    content = f"```python\n{content}\n```"

            yield gr.ChatMessage(
                role="assistant",
                content=content,
                metadata={
                    "title": f"🛠️ Used tool {first_tool_call.name}",
                    "id": f"call_{len(step_log.tool_calls)}",
                    "status": "done",
                },
            )

        if hasattr(step_log, "observations") and step_log.observations and step_log.observations.strip():
            log_content = re.sub(r"^Execution logs:\s*",
                                 "", step_log.observations.strip())
            yield gr.ChatMessage(
                role="assistant",
                content=f"```bash\n{log_content}\n```",
                metadata={"title": "📝 Execution Logs", "status": "done"},
            )

        if hasattr(step_log, "error") and step_log.error:
            yield gr.ChatMessage(
                role="assistant",
                content=str(step_log.error),
                metadata={"title": "💥 Error", "status": "done"},
            )

        if getattr(step_log, "observations_images", []):
            for image in step_log.observations_images:
                path_image = AgentImage(image).to_string()
                yield gr.ChatMessage(
                    role="assistant",
                    content={"path": path_image,
                             "mime_type": f"image/{path_image.split('.')[-1]}"},
                    metadata={"title": "🖼️ Output Image", "status": "done"},
                )

        yield gr.ChatMessage(role="assistant", content=get_step_footnote_content(step_log, step_number))
        yield gr.ChatMessage(role="assistant", content="-----", metadata={"status": "done"})

    elif isinstance(step_log, PlanningStep):
        yield gr.ChatMessage(role="assistant", content="**Planning step**")
        yield gr.ChatMessage(role="assistant", content=step_log.plan)
        yield gr.ChatMessage(
            role="assistant",
            content=get_step_footnote_content(step_log, "Planning step")
        )
        yield gr.ChatMessage(role="assistant", content="-----", metadata={"status": "done"})

    elif isinstance(step_log, FinalAnswerStep):
        final_answer = step_log.final_answer
        if isinstance(final_answer, AgentText):
            yield gr.ChatMessage(
                role="assistant",
                content=f"**Final answer:**\n{final_answer.to_string()}\n",
            )
        elif isinstance(final_answer, AgentImage):
            yield gr.ChatMessage(
                role="assistant",
                content={"path": final_answer.to_string(),
                         "mime_type": "image/png"},
            )
        elif isinstance(final_answer, AgentAudio):
            yield gr.ChatMessage(
                role="assistant",
                content={"path": final_answer.to_string(),
                         "mime_type": "audio/wav"},
            )
        else:
            yield gr.ChatMessage(
                role="assistant",
                content=f"**Final answer:** {str(final_answer)}"
            )
    else:
        raise ValueError(f"Unsupported step type: {type(step_log)}")


def stream_to_gradio(
    agent,
    task: str,
    reset_agent_memory: bool = False,
    additional_args: Optional[dict] = None,
):
    """Runs an agent with the given task and streams the messages from the agent as gradio ChatMessages."""
    total_input_tokens = 0
    total_output_tokens = 0

    for step_log in agent.run(task, stream=True, reset=reset_agent_memory, additional_args=additional_args):
        if getattr(agent.model, "last_input_token_count", None) is not None:
            total_input_tokens += agent.model.last_input_token_count
            total_output_tokens += agent.model.last_output_token_count
            if isinstance(step_log, (ActionStep, PlanningStep)):
                step_log.input_token_count = agent.model.last_input_token_count
                step_log.output_token_count = agent.model.last_output_token_count

        for message in pull_messages_from_step(step_log):
            yield message


class GradioUI:
    """A one-line interface to launch your agent in Gradio"""

    def __init__(self, agent: MultiStepAgent, file_upload_folder: str | None = None):
        if not _is_package_available("gradio"):
            raise ModuleNotFoundError(
                "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
            )
        self.agent = agent
        self.file_upload_folder = file_upload_folder
        self.name = getattr(agent, "name") or "Agent interface"
        self.description = getattr(agent, "description", None)
        if self.file_upload_folder is not None and not os.path.exists(file_upload_folder):
            os.mkdir(file_upload_folder)

    def interact_with_agent(self, prompt, messages, session_state):
        import gradio as gr

        if "agent" not in session_state:
            session_state["agent"] = self.agent

        try:
            messages.append(gr.ChatMessage(role="user", content=prompt))
            yield messages

            for msg in stream_to_gradio(session_state["agent"], task=prompt, reset_agent_memory=False):
                messages.append(msg)
                yield messages

            yield messages
        except Exception as e:
            messages.append(gr.ChatMessage(
                role="assistant", content=f"Error: {str(e)}"))
            yield messages

    def upload_file(self, file, file_uploads_log, allowed_file_types=None):
        import gradio as gr

        if file is None:
            return gr.Textbox(value="No file uploaded", visible=True), file_uploads_log

        if allowed_file_types is None:
            allowed_file_types = [".pdf", ".docx", ".txt"]

        file_ext = os.path.splitext(file.name)[1].lower()
        if file_ext not in allowed_file_types:
            return gr.Textbox("File type disallowed", visible=True), file_uploads_log

        original_name = os.path.basename(file.name)
        sanitized_name = re.sub(r"[^\w\-.]", "_", original_name)
        file_path = os.path.join(self.file_upload_folder, sanitized_name)
        shutil.copy(file.name, file_path)

        return gr.Textbox(f"File uploaded: {file_path}", visible=True), file_uploads_log + [file_path]

    def log_user_message(self, text_input, file_uploads_log):
        import gradio as gr

        return (
            text_input
            + (
                f"\nYou have been provided with these files: {file_uploads_log}"
                if file_uploads_log else ""
            ),
            "",
            gr.Button(interactive=False),
        )

    def launch(self, share: bool = True, **kwargs):
        self.create_app().launch(debug=True, share=share, **kwargs)

    def create_app(self):
        import gradio as gr

        with gr.Blocks(css=CUSTOM_CSS, theme="ocean", fill_height=True) as demo:
            create_header()

            session_state = gr.State({})
            stored_messages = gr.State([])
            file_uploads_log = gr.State([])

            # Main content area: Chat + Input
            with gr.Row(equal_height=True, variant="panel", elem_classes="content-wrap"):
                # Column for chat and input
                with gr.Column(scale=3):
                    # Input area moved here
                    # gr.Markdown("**Your request**")
                    text_input = gr.Textbox(
                        lines=2,
                        label="Your request",
                        placeholder="Enter your prompt here and press Shift+Enter or the button",
                    )
                    submit_btn = gr.Button(
                        "Submit", variant="primary", elem_classes="full-width-btn"
                    )

                    # Chatbot
                    chatbot = gr.Chatbot(
                        label="Agent",
                        type="messages",
                        avatar_images=(
                            None,
                            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
                        ),
                        resizeable=True,
                        scale=1,
                    )

                # Optional: Column for file uploads
                if self.file_upload_folder is not None:
                    with gr.Column(scale=1):
                        gr.Markdown("**Upload Files**")
                        upload_file = gr.File(label="Upload a file")
                        upload_status = gr.Textbox(
                            label="Upload Status", interactive=False, visible=False
                        )
                        upload_file.change(
                            self.upload_file,
                            [upload_file, file_uploads_log],
                            [upload_status, file_uploads_log],
                        )

                # Wiring interactions
                text_input.submit(
                    self.log_user_message,
                    [text_input, file_uploads_log],
                    [stored_messages, text_input, submit_btn],
                ).then(
                    self.interact_with_agent,
                    [stored_messages, chatbot, session_state],
                    [chatbot],
                ).then(
                    lambda: (
                        gr.update(value="", interactive=True,
                                  placeholder="Enter your prompt here and press Shift+Enter or the button"),
                        gr.update(interactive=True),
                    ),
                    None,
                    [text_input, submit_btn],
                )

                submit_btn.click(
                    self.log_user_message,
                    [text_input, file_uploads_log],
                    [stored_messages, text_input, submit_btn],
                ).then(
                    self.interact_with_agent,
                    [stored_messages, chatbot, session_state],
                    [chatbot],
                ).then(
                    lambda: (
                        gr.update(value="", interactive=True,
                                  placeholder="Enter your prompt here and press Shift+Enter or the button"),
                        gr.update(interactive=True),
                    ),
                    None,
                    [text_input, submit_btn],
                )

            create_footer()

        return demo


__all__ = ["stream_to_gradio", "GradioUI"]
