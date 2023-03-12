import gradio as gr
from visvl.textvqa_tab import textvqa_tab_context
from visvl.textcaps_tab import textcaps_tab_context
from visvl.consts import CSS_FORMAT

def launch_gradio_app():

    with gr.Blocks(css=CSS_FORMAT) as demo:

        textvqa_tab_context()

        textcaps_tab_context()

    demo.launch(share=True)

if __name__ == '__main__':
    launch_gradio_app()
