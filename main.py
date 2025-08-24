import os
import random
from datetime import datetime

import chatbot.chatbot_assistant as chatbot_assistant
import psutil
import streamlit as st
from streamlit.components.v1 import html

MODEL_PATH = "assistant.pth"
DIMENSIONS_PATH = "dimensions.json"
STYLE_PATH = "css/style.css"
SCRIPT_PATH = "js/script.js"

if __name__ == "__main__":

    def get_time():
        current_time = datetime.now().strftime("%I:%M %p")
        return f"It is {current_time}"

    def get_date():
        current_date = datetime.now().strftime("%Y-%m-%d")
        return f"Today's date is {current_date}"

    def get_stock():
        stocks = ["AAPL", "GOOGL", "MSFT"]
        return random.choice(stocks)

    if not os.path.exists(MODEL_PATH):
        assistant = chatbot_assistant.ChatbotAssistant(
            "intents.json", method_mappings={"get_time": get_time, "get_date": get_date, "get_stock": get_stock}
        )
        assistant.parse_intents()
        assistant.prepare_data()
        assistant.train_model(batch_size=8, lr=0.001, epochs=100)

        assistant.save_model(MODEL_PATH, DIMENSIONS_PATH)
    else:
        assistant = chatbot_assistant.ChatbotAssistant(
            "intents.json", method_mappings={"get_time": get_time, "get_date": get_date, "get_stock": get_stock}
        )
        assistant.parse_intents()
        assistant.load_model(MODEL_PATH, DIMENSIONS_PATH)

    if assistant:
        with open(STYLE_PATH) as f:
            css = f.read()
        with open(SCRIPT_PATH) as f:
            js = f.read()

        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
        html(f"<script>{js}</script>")

        st.title("Chatbot")
        st.markdown(
            "Enter the message to start communicating with the Chatbot. Your messages are displayed on the right, and the Chatbot's answers are on the left:",
            unsafe_allow_html=True,
        )
        st.markdown("<div id='messages'></div>", unsafe_allow_html=True)

        with st.form("chatbot", clear_on_submit=True, border=False):
            message = st.text_input(
                "Input field",
                placeholder="Enter your message",
                key="chatbot_input",
                disabled=False,
                label_visibility="hidden",
            )
            submit = st.form_submit_button("Send")

        quit_app = st.button("Quit App")

        if quit_app:
            pid = os.getpid()
            p = psutil.Process(pid)
            p.terminate()

        if submit and message:
            html(
                f'<script>window.parent.document.displayMessages("{message}", "{assistant.process_message(message)}");</script>'
            )
        else:
            st.stop()
