# gradio_setup.py

from typing import Iterator
from decouple import config
import gradio as gr
from llama2_wrapper import LLAMA2_WRAPPER
from .config import Config
from .business_logic import generate_message, process_example, check_input_token_length

def setup_gradio_ui():
    DESCRIPTION = """
    # llama2-webui
    This is a chatbot based on Llama-2. 
    """
    DESCRIPTION2 = """
    - Model: 
        [Llama 2 7B Chat - GGUF](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF)
    - Supporting model backends: 
        [transformers](https://github.com/huggingface/transformers), 
        [bitsandbytes(8-bit inference)](https://github.com/TimDettmers/bitsandbytes), 
        [AutoGPTQ(4-bit inference)](https://github.com/PanQiWei/AutoGPTQ), 
        [llama.cpp](https://github.com/ggerganov/llama.cpp)"""

    def clear_and_save_textbox(message: str) -> tuple[str, str]:
        return "", message

    def display_input(
        message: str, history: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        history.append((message, ""))
        return history

    def delete_prev_fn(
        history: list[tuple[str, str]]
    ) -> tuple[list[tuple[str, str]], str]:
        try:
            message, _ = history.pop()
        except IndexError:
            message = ""
        return history, message or ""
    
    with gr.Blocks() as demo:
        gr.Markdown(DESCRIPTION)

        # Define your Gradio UI components and their interactions here
        with gr.Group():
            chatbot = gr.Chatbot(label="Chatbot")
            with gr.Row():
                textbox = gr.Textbox(
                    container=False,
                    show_label=False,
                    placeholder="Type a message...",
                    scale=10,
                )
                submit_button = gr.Button(
                    "Submit", variant="primary", scale=1, min_width=0
                )
        with gr.Row():
            retry_button = gr.Button("🔄  Retry", variant="secondary")
            undo_button = gr.Button("↩️ Undo", variant="secondary")
            clear_button = gr.Button("🗑️  Clear", variant="secondary")

        saved_input = gr.State()

        with gr.Accordion(label="Advanced options", open=False):
            system_prompt = gr.Textbox(
                label="System prompt", value=Config.DEFAULT_SYSTEM_PROMPT, lines=6
            )
            max_new_tokens = gr.Slider(
                label="Max new tokens",
                minimum=1,
                maximum=Config.MAX_MAX_NEW_TOKENS,
                step=1,
                value=Config.DEFAULT_MAX_NEW_TOKENS,
            )
            temperature = gr.Slider(
                label="Temperature",
                minimum=0.1,
                maximum=4.0,
                step=0.1,
                value=1.0,
            )
            top_p = gr.Slider(
                label="Top-p (nucleus sampling)",
                minimum=0.05,
                maximum=1.0,
                step=0.05,
                value=0.95,
            )
            top_k = gr.Slider(
                label="Top-k",
                minimum=1,
                maximum=1000,
                step=1,
                value=50,
            )

        gr.Examples(
            examples=[
                "Hello there! How are you doing?",
                "Can you explain briefly to me what is the Python programming language?",
                "Explain the plot of Cinderella in a sentence.",
                "How many hours does it take a man to eat a Helicopter?",
                "Write a 100-word article on 'Benefits of Open-Source in AI research'",
            ],
            inputs=textbox,
            outputs=[textbox, chatbot],
            fn=process_example,
            cache_examples=True,
        )

        gr.Markdown(DESCRIPTION2)

        textbox.submit(
            fn=clear_and_save_textbox,
            inputs=textbox,
            outputs=[textbox, saved_input],
            api_name=False,
            queue=False,
        ).then(
            fn=display_input,
            inputs=[saved_input, chatbot],
            outputs=chatbot,
            api_name=False,
            queue=False,
        ).then(
            fn=check_input_token_length,
            inputs=[saved_input, chatbot, system_prompt],
            api_name=False,
            queue=False,
        ).success(
            fn=generate_message,
            inputs=[
                saved_input,
                chatbot,
                system_prompt,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
            ],
            outputs=chatbot,
            api_name=False,
        )

        button_event_preprocess = (
            submit_button.click(
                fn=clear_and_save_textbox,
                inputs=textbox,
                outputs=[textbox, saved_input],
                api_name=False,
                queue=False,
            )
            .then(
                fn=display_input,
                inputs=[saved_input, chatbot],
                outputs=chatbot,
                api_name=False,
                queue=False,
            )
            .then(
                fn=check_input_token_length,
                inputs=[saved_input, chatbot, system_prompt],
                api_name=False,
                queue=False,
            )
            .success(
                fn=generate_message,
                inputs=[
                    saved_input,
                    chatbot,
                    system_prompt,
                    max_new_tokens,
                    temperature,
                    top_p,
                    top_k,
                ],
                outputs=chatbot,
                api_name=False,
            )
        )

        retry_button.click(
            fn=delete_prev_fn,
            inputs=chatbot,
            outputs=[chatbot, saved_input],
            api_name=False,
            queue=False,
        ).then(
            fn=display_input,
            inputs=[saved_input, chatbot],
            outputs=chatbot,
            api_name=False,
            queue=False,
        ).then(
            fn=generate_message,
            inputs=[
                saved_input,
                chatbot,
                system_prompt,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
            ],
            outputs=chatbot,
            api_name=False,
        )

        undo_button.click(
            fn=delete_prev_fn,
            inputs=chatbot,
            outputs=[chatbot, saved_input],
            api_name=False,
            queue=False,
        ).then(
            fn=lambda x: x,
            inputs=[saved_input],
            outputs=textbox,
            api_name=False,
            queue=False,
        )

        clear_button.click(
            fn=lambda: ([], ""),
            outputs=[chatbot, saved_input],
            queue=False,
            api_name=False,
        )

    demo.queue(max_size=20).launch()

if __name__ == "__main__":
    setup_gradio_ui()

