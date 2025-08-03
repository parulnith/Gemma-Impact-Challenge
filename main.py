from ui import main_ui

if __name__ == "__main__":
    ui = main_ui()
    ui.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)
