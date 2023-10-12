# src/chatbot_ui/gradio_interface.py

from typing import Iterator
from decouple import config
import gradio as gr
from llama2_wrapper import LLAMA2_WRAPPER
from .config import Config


# Initialize the llama2_wrapper module using configuration settings from config.py
print(f"Initializing the llama2_wrapper module")

llama2_wrapper = LLAMA2_WRAPPER(
    model_path=Config.MODEL_PATH,
    backend_type=Config.BACKEND_TYPE,
    max_tokens=Config.MAX_INPUT_TOKEN_LENGTH,
    load_in_8bit=Config.LOAD_IN_8BIT,
)


def start_gradio():
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

    def generate(
        generate_message: str,
        generate_history_with_input: list[tuple[str, str]],
        generate_system_prompt: str,
        generate_max_new_tokens: int,
        generate_temperature: float,
        generate_top_p: float,
        generate_top_k: int,
    ) -> Iterator[list[tuple[str, str]]]:
        if generate_max_new_tokens > Config.MAX_MAX_NEW_TOKENS:
            raise ValueError

        history = generate_history_with_input[:-1]
        generator = llama2_wrapper.run(
            generate_message, history, generate_system_prompt, generate_max_new_tokens, generate_temperature, generate_top_p, generate_top_k
        )
        try:
            first_response = next(generator)
            yield history + [(generate_message, first_response)]
        except StopIteration:
            yield history + [(generate_message, "")]
        for response in generator:
            yield history + [(generate_message, response)]

    def process_example(message: str) -> tuple[str, list[tuple[str, str]]]:
        generator = generate(message, [], Config.DEFAULT_SYSTEM_PROMPT, 1024, 1, 0.95, 50)
        x = []  # Initialize x with an empty list as a default value
        for x_item in generator:
            x = x_item  # Update x with the last value yielded by the generator
        return "", x

    def check_input_token_length(
        message: str, chat_history: list[tuple[str, str]], token_length_system_prompt: str
    ) -> None:
        input_token_length = llama2_wrapper.get_input_token_length(
            message, chat_history, token_length_system_prompt
        )
        if input_token_length > Config.MAX_INPUT_TOKEN_LENGTH:
            raise gr.Error(
                f"The accumulated input is too long ({input_token_length} > {Config.MAX_INPUT_TOKEN_LENGTH}). Clear your chat history and try again."
            )

    with gr.Blocks() as demo:
        gr.Markdown(DESCRIPTION)

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
            fn=generate,
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
                fn=generate,
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
            fn=generate,
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
    start_gradio()
