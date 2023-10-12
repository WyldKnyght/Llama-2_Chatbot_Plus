# src/main.py

from chatbot_ui.gradio_setup import setup_gradio_ui
from chatbot_ui.business_logic import generate_message, process_example, check_input_token_length

if __name__ == "__main__":
    setup_gradio_ui()

