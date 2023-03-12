import gradio as gr
from .consts import TEXTVQA_VIS_ROOT, NUM_CONTRAST
from .utils import get_file_list, random_select_textvqa, get_sample_textvqa

def textvqa_tab_context():

    textvqa_dropdown_list = []
    textvqa_dropdown_list_default = []
    textvqa_vis_files_list = get_file_list(TEXTVQA_VIS_ROOT)
    textvqa_vis_blip2_withocr_files_list = [item for item in textvqa_vis_files_list if item[:11] == 'BLIP2_With_']
    textvqa_vis_blip2_withoutocr_files_list = [item for item in textvqa_vis_files_list if item[:11] == 'BLIP2_Witho']
    textvqa_vis_m4c_files_list = [item for item in textvqa_vis_files_list if item[:3] == 'M4C']

    textvqa_dropdown_list.extend([textvqa_vis_blip2_withoutocr_files_list, \
         textvqa_vis_blip2_withocr_files_list, textvqa_vis_m4c_files_list])
    textvqa_dropdown_list_default.extend([1, 5, 0])

    textvqa_template_contrasts_list = []
    textvqa_prediction_contrasts_list = []

    with gr.Tab("TextVQA"):

        with gr.Row():

            with gr.Row():
                textvqa_split_radio = gr.Radio(label='Split', choices=['val', 'test'], value='val')
                textvqa_random_select_button = gr.Button(value="Random Select", variant="primary")
            textvqa_sample_index_slider = gr.Slider(label='Index', minimum=1, maximum=5734, step=1)

        with gr.Row():

            textvqa_image_input = gr.Image(label='Image')

            with gr.Column():

                with gr.Box(elem_id=f"textvqa_anno_box"):

                    textvqa_ocr_input = gr.Textbox(label='OCR Texts')
                    textvqa_question_input = gr.Textbox(label='Question')
                    textvqa_answer_output_gt = gr.Textbox(label='10 GroundTruth Answers')

                for i in range(NUM_CONTRAST):
                    # Prompt Template Dropdown and prediction text output
                    with gr.Box(elem_id=f"textvqa_contrast_box{i+1}"):

                        textvqa_template_contrasts_list.append(gr.Dropdown(label=f"Prompt Template {i+1}", \
                            choices=textvqa_dropdown_list[i], multiselect=False, \
                                value=textvqa_dropdown_list[i][textvqa_dropdown_list_default[i]]))
                        textvqa_prediction_contrasts_list.append(gr.Textbox(label=f"Predicted Answer {i+1}"))

    textvqa_random_select_button.click(fn=random_select_textvqa, \
                                       inputs=[textvqa_split_radio, \
                                         *textvqa_template_contrasts_list], \
                                       outputs=[textvqa_image_input, textvqa_ocr_input, \
                                         textvqa_question_input, textvqa_answer_output_gt, \
                                         *textvqa_prediction_contrasts_list, \
                                         textvqa_sample_index_slider])

    textvqa_sample_index_slider.change(get_sample_textvqa, \
                                       inputs=[textvqa_split_radio, textvqa_sample_index_slider, \
                                        *textvqa_template_contrasts_list], \
                                       outputs=[textvqa_image_input, textvqa_ocr_input, \
                                         textvqa_question_input, textvqa_answer_output_gt, \
                                         *textvqa_prediction_contrasts_list])