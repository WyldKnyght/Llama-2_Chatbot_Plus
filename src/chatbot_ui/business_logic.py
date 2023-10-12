# business_logic.py

from typing import Iterator
from llama2_wrapper import LLAMA2_WRAPPER
from .config import Config
import gradio as gr

# Initialize the llama2_wrapper module using configuration settings from config.py
llama2_wrapper = LLAMA2_WRAPPER(
    model_path=Config.MODEL_PATH,
    backend_type=Config.BACKEND_TYPE,
    max_tokens=Config.MAX_INPUT_TOKEN_LENGTH,
    load_in_8bit=Config.LOAD_IN_8BIT,
)

def generate_message(message, history_with_input, system_prompt, max_new_tokens, temperature, top_p, top_k):
    history = history_with_input[:-1]
    generator = llama2_wrapper.run(
        message, history, system_prompt, max_new_tokens, temperature, top_p, top_k
    )
    try:
        first_response = next(generator)
        yield history + [(message, first_response)]
    except StopIteration:
        yield history + [(message, "")]
    for response in generator:
        yield history + [(message, response)]

def process_example(message):
    generator = generate_message(message, [], Config.DEFAULT_SYSTEM_PROMPT, 1024, 1, 0.95, 50)
    x = []  # Initialize x with an empty list as a default value
    for x_item in generator:
        x = x_item  # Update x with the last value yielded by the generator
    return "", x

def check_input_token_length(message, chat_history, token_length_system_prompt):
    input_token_length = llama2_wrapper.get_input_token_length(
        message, chat_history, token_length_system_prompt
    )
    if input_token_length > Config.MAX_INPUT_TOKEN_LENGTH:
        raise gr.Error(
            f"The accumulated input is too long ({input_token_length} > {Config.MAX_INPUT_TOKEN_LENGTH}). Clear your chat history and try again."
        )
