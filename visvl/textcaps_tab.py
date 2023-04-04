import re
import gradio as gr
from .consts import TEXTCAPS_VIS_ROOT
from .utils import get_file_list, random_select_textcaps, get_sample_textcaps

def textcaps_tab_context():

    textcaps_dropdown_list = []
    textcaps_dropdown_list_default = []
    textcaps_vis_files_list = get_file_list(TEXTCAPS_VIS_ROOT)
    textcaps_pattern_list = [
        # Withoutocr_zeroshot
        ("^BLIP2_Without.*/zeroshot.*", 1),
        # Withocr_zeroshot
        ("^BLIP2_With_.*/zeroshot.*", 7),
        # Withoutocr_fewshot
        # ("^BLIP2_Without.*/fewshot.*", 0),
        # Withocr_fewshot
        ("^BLIP2_With_.*/fewshot.*", 0),
        # Withoutocr_fullshot
        ("^BLIP2_Without.*/fullshot.*", 0),
        # Withocr_fullshot
        ("^BLIP2_With_.*/fullshot.*", 0),
    ]
    NUM_CONTRAST = len(textcaps_pattern_list)
    for pattern, default_index in textcaps_pattern_list:
        textcaps_dropdown_list.append([item for item in textcaps_vis_files_list if re.search(pattern, item)])
        textcaps_dropdown_list_default.append(default_index)


    textcaps_template_contrasts_list = []
    textcaps_prediction_contrasts_list = []

    with gr.Tab("TextCaps"):

        with gr.Row():

            with gr.Row():
                textcaps_split_radio = gr.Radio(label='Split', choices=['val', 'test'], value='val')
                textcaps_random_select_button = gr.Button(value="Random Select", variant="primary")
            textcaps_sample_index_slider = gr.Slider(label='Index', minimum=1, maximum=3289, step=1)

        with gr.Row():

            textcaps_image_input = gr.Image(label='Image')

            with gr.Column():

                with gr.Box(elem_id=f"textcaps_anno_box"):

                    textcaps_ocr_input = gr.Textbox(label='OCR texts')
                    textcaps_caption_output_gt = gr.Textbox(label='5 GroundTruth Captions')

                for i in range(NUM_CONTRAST):
                    with gr.Box(elem_id=f"textcaps_contrast_box{i+1}"):
                        textcaps_template_contrasts_list.append(gr.Dropdown(label=f"Prompt Template {i+1}", \
                             choices=textcaps_dropdown_list[i], \
                                value=textcaps_dropdown_list[i][textcaps_dropdown_list_default[i]], multiselect=False))
                        textcaps_prediction_contrasts_list.append(gr.Textbox(label=f"Predicted Caption {i+1}"))


    textcaps_random_select_button.click(fn=random_select_textcaps, \
                                       inputs=[textcaps_split_radio, *textcaps_template_contrasts_list], \
                                       outputs=[textcaps_image_input, textcaps_ocr_input, \
                                           textcaps_caption_output_gt, *textcaps_prediction_contrasts_list,\
                                            textcaps_sample_index_slider])

    textcaps_sample_index_slider.change(get_sample_textcaps, \
                                       inputs=[textcaps_split_radio, textcaps_sample_index_slider, \
                                        *textcaps_template_contrasts_list], \
                                       outputs=[textcaps_image_input, textcaps_ocr_input, \
                                           textcaps_caption_output_gt, \
                                           *textcaps_prediction_contrasts_list])
