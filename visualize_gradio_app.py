from pathlib import Path
import json
import random
import gradio as gr

# CONSTANT DEFINITION
TEXTVQA_VIS_ROOT = '/home/mrxir/Codes/textvl.vis/textvqa/'
TEXTCAPS_VIS_ROOT = '/home/mrxir/Codes/textvl.vis/textcaps/'
TEXTVL_IMAGE_ROOT = '/home/mrxir/Codes/textvl.vis/TextVL/Images/'
TEXTVQA_ANNOTATION_ROOT = '/home/mrxir/Codes/textvl.vis/TextVL/Annotations/VQA/'
TEXTCAPS_ANNOTATION_ROOT = '/home/mrxir/Codes/textvl.vis/TextVL/Annotations/IC/'
TEXTVL_OCR_ANNOTATION_ROOT = '/home/mrxir/Codes/textvl.vis/TextVL/Annotations/OCR/'

def get_file_list(root_path):
    return sorted(list(set(['/'.join(str(item).split('/')[-3:-1]) for item in Path(root_path).glob('**/*') if item.is_file()])))

def dict_list2dict_dict(dict_list, key_name):
    dict_dict = dict()
    for item in dict_list:
        if item[key_name] not in dict_dict:
            dict_dict[item[key_name]] = dict()
        dict_dict[item[key_name]] = item
    return dict_dict

textvqa_vis_files_list = get_file_list(TEXTVQA_VIS_ROOT)
textcaps_vis_files_list = get_file_list(TEXTCAPS_VIS_ROOT)

def load_data_textvqa(textvqa_sample_split, textvqa_prompt_vis_subdir):

    global textvqa_samples_data, textvqa_annotation_data, textvl_ocr_data
    global textvqa_annotation_data_dict, textvl_ocr_data_dict

    textvqa_vis_file_path = Path(TEXTVQA_VIS_ROOT)/textvqa_prompt_vis_subdir/f"{textvqa_sample_split}_vqa_result.json"
    textvqa_anno_file_path = Path(TEXTVQA_ANNOTATION_ROOT)/f"TextVQA_0.5.1_{textvqa_sample_split}.json"
    textvl_ocr_file_path = Path(TEXTVL_OCR_ANNOTATION_ROOT)/f"TextVQA_Rosetta_OCR_v0.2_{textvqa_sample_split}.json"

    textvqa_samples_data = json.load(textvqa_vis_file_path.open())
    textvqa_annotation_data = json.load(textvqa_anno_file_path.open())['data']
    textvqa_annotation_data_dict = dict_list2dict_dict(textvqa_annotation_data, 'question_id')
    textvl_ocr_data = json.load(textvl_ocr_file_path.open())['data']
    textvl_ocr_data_dict = dict_list2dict_dict(textvl_ocr_data, 'image_id')

def load_data_textcaps(textcaps_sample_split, textcaps_prompt_vis_subdir):

    global textcaps_samples_data, textcaps_annotation_data, textvl_ocr_data
    global textcaps_annotation_data_dict, textvl_ocr_data_dict

    textcaps_vis_file_path = Path(TEXTCAPS_VIS_ROOT)/textcaps_prompt_vis_subdir/f"{textcaps_sample_split}_epochbest.json"
    textcaps_anno_file_path = Path(TEXTCAPS_ANNOTATION_ROOT)/f"TextCaps_0.1_{textcaps_sample_split}.json"
    textvl_ocr_file_path = Path(TEXTVL_OCR_ANNOTATION_ROOT)/f"TextVQA_Rosetta_OCR_v0.2_{textcaps_sample_split}.json"

    textcaps_samples_data = json.load(textcaps_vis_file_path.open())
    textcaps_annotation_data = json.load(textcaps_anno_file_path.open())['data']
    textcaps_annotation_data_dict = dict_list2dict_dict(textcaps_annotation_data, 'image_id')
    textvl_ocr_data = json.load(textvl_ocr_file_path.open())['data']
    textvl_ocr_data_dict = dict_list2dict_dict(textvl_ocr_data, 'image_id')

def random_select_textvqa(split):

    index = random.randint(1, len(textvqa_samples_data))
    sample = get_sample_textvqa(index, split)
    sample.append(index)
    return sample

def random_select_textcaps(split):

    index = random.randint(1, len(textcaps_samples_data))
    sample = get_sample_textcaps(index, split)
    sample.append(index)
    return sample

def get_sample_textvqa(index, split):

    sample_data = textvqa_samples_data[index]
    question_id = sample_data['question_id']
    answer_output_pred = sample_data['answer']

    question_anno_data = textvqa_annotation_data_dict[question_id]
    image_id = question_anno_data['image_id']
    question_input = question_anno_data['question']
    answer_output_gt = '\n'.join(question_anno_data['answers'])
    ocr_input = ' '.join(textvl_ocr_data_dict[image_id]['ocr_tokens'])
    image_split = 'trainval' if split == 'val' else 'test'
    image_input = Path(TEXTVL_IMAGE_ROOT)/image_split/f"{image_id}.jpg"
    
    return [str(image_input), ocr_input, question_input, answer_output_pred, answer_output_gt]

def get_sample_textcaps(index, split):

    sample_data = textcaps_samples_data[index]
    image_id = sample_data['image_id']
    caption_output_pred = sample_data['caption']

    caption_anno_data = textcaps_annotation_data_dict[image_id]
    caption_output_gt = '\n'.join(caption_anno_data['reference_strs'])
    ocr_input = ' '.join(textvl_ocr_data_dict[image_id]['ocr_tokens'])
    image_split = 'trainval' if split == 'val' else 'test'
    image_input = Path(TEXTVL_IMAGE_ROOT)/image_split/f"{image_id}.jpg"

    return [str(image_input), ocr_input, caption_output_pred, caption_output_gt]


def launch_gradio_app():

    with gr.Blocks() as demo:

        # gr.Markdown("# TextVQA and TextCaps Result Visualization Demo")

        with gr.Tab("TextVQA"):

            with gr.Row():

                with gr.Row():
                    textvqa_split_radio = gr.Radio(label='TextVQA Sample Split', choices=['val', 'test'], value='val')
                    textvqa_random_select_button = gr.Button(value="Random Select", variant="primary")
                textvqa_prompt_template_dropdown = gr.Dropdown(label='TextVQA Prompt Template', \
                     choices=textvqa_vis_files_list, multiselect=False)

            with gr.Row():

                textvqa_image_input = gr.Image(label='TextVQA Sample Image Input')

                with gr.Column():

                    textvqa_sample_index_slider = gr.Slider(label='TextVQA Sample Index', minimum=1, maximum=5734, step=1)
                    textvqa_ocr_input = gr.Textbox(label='TextVQA Sample OCR Input')
                    textvqa_question_input = gr.Textbox(label='TextVQA Sample Question Input')
                    textvqa_answer_output_pred = gr.Textbox(label='TextVQA Sample Answer Prediction Output')
                    textvqa_answer_output_gt = gr.Textbox(label='TextVQA Sample Answer GroundTruth Outputs')

        textvqa_random_select_button.click(fn=random_select_textvqa, inputs=textvqa_split_radio, \
                                           outputs=[textvqa_image_input, textvqa_ocr_input, \
                                               textvqa_question_input, textvqa_answer_output_pred, \
                                                 textvqa_answer_output_gt, textvqa_sample_index_slider])

        textvqa_prompt_template_dropdown.change(load_data_textvqa, \
            inputs=[textvqa_split_radio, textvqa_prompt_template_dropdown])

        textvqa_sample_index_slider.change(get_sample_textvqa, \
                                           inputs=[textvqa_sample_index_slider, textvqa_split_radio], \
                                           outputs=[textvqa_image_input, textvqa_ocr_input, \
                                               textvqa_question_input, textvqa_answer_output_pred, \
                                                 textvqa_answer_output_gt])

        with gr.Tab("TextCaps"):

            with gr.Row():

                with gr.Row():
                    textcaps_split_radio = gr.Radio(label='TextCaps Sample Split', choices=['val', 'test'], value='val')
                    textcaps_random_select_button = gr.Button(value="Random Select", variant="primary")
                textcaps_prompt_template_dropdown = gr.Dropdown(label='TextCaps Prompt Template', \
                     choices=textcaps_vis_files_list, multiselect=False)

            with gr.Row():

                textcaps_image_input = gr.Image(label='TextCaps Sample Image Input')

                with gr.Column():

                    textcaps_sample_index_slider = gr.Slider(label='TextCaps Sample Index', minimum=1, maximum=3289, step=1)
                    textcaps_ocr_input = gr.Textbox(label='TextCaps Sample OCR Input')
                    textcaps_caption_output_pred = gr.Textbox(label='TextCaps Sample Caption Prediction Output')
                    textcaps_caption_output_gt = gr.Textbox(label='TextCaps Sample Caption GroundTruth Output')

        textcaps_random_select_button.click(fn=random_select_textcaps, inputs=textcaps_split_radio, \
                                           outputs=[textcaps_image_input, textcaps_ocr_input, \
                                               textcaps_caption_output_pred, textcaps_caption_output_gt, \
                                                textcaps_sample_index_slider])

        textcaps_prompt_template_dropdown.change(load_data_textcaps, \
            inputs=[textcaps_split_radio, textcaps_prompt_template_dropdown])

        textcaps_sample_index_slider.change(get_sample_textcaps, \
                                           inputs=[textcaps_sample_index_slider, textcaps_split_radio], \
                                           outputs=[textcaps_image_input, textcaps_ocr_input, \
                                               textcaps_caption_output_pred, textcaps_caption_output_gt])

    demo.launch(share=True)

if __name__ == '__main__':
    _, _, public_link = launch_gradio_app()
    print(f"=> public_link: {public_link}")
