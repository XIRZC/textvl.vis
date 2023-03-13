from pathlib import Path
import json
import random
from .consts import *

def get_file_list(root_path):
    return sorted(list(set(['/'.join(str(item).split('/')[-3:-1]) for item in Path(root_path).glob('**/*') if item.is_file()])))

def dict_list2dict_dict(dict_list, key_name):
    dict_dict = dict()
    for item in dict_list:
        if item[key_name] not in dict_dict:
            dict_dict[item[key_name]] = dict()
        dict_dict[item[key_name]] = item
    return dict_dict

def load_data_textvl_ocr(split):

    textvl_ocr_file_path = Path(TEXTVL_OCR_ANNOTATION_ROOT)/f"TextVQA_Rosetta_OCR_v0.2_{split}.json"

    textvl_ocr_data = json.load(textvl_ocr_file_path.open())['data']
    textvl_ocr_data_dict = dict_list2dict_dict(textvl_ocr_data, 'image_id')
    
    return textvl_ocr_data_dict

def load_anno_textvqa(textvqa_sample_split):

    textvqa_anno_file_path = Path(TEXTVQA_ANNOTATION_ROOT)/f"TextVQA_0.5.1_{textvqa_sample_split}.json"

    textvqa_anno_data = json.load(textvqa_anno_file_path.open())['data']
    textvqa_anno_data_dict = dict_list2dict_dict(textvqa_anno_data, 'question_id')

    return textvqa_anno_data_dict

def load_anno_textcaps(textcaps_sample_split):

    textcaps_anno_file_path = Path(TEXTCAPS_ANNOTATION_ROOT)/f"TextCaps_0.1_{textcaps_sample_split}.json"

    textcaps_anno_data = json.load(textcaps_anno_file_path.open())['data']
    textcaps_anno_data_dict = dict_list2dict_dict(textcaps_anno_data, 'image_id')
    
    return textcaps_anno_data_dict

def load_samples_textvqa(textvqa_sample_split, textvqa_prompt_vis_subdir):

    textvqa_vis_file_path = Path(TEXTVQA_VIS_ROOT)/textvqa_prompt_vis_subdir/f"{textvqa_sample_split}_vqa_result.json"

    textvqa_samples_data = json.load(textvqa_vis_file_path.open())

    return textvqa_samples_data

def load_samples_textcaps(textcaps_sample_split, textcaps_prompt_vis_subdir):

    textcaps_vis_file_path = Path(TEXTCAPS_VIS_ROOT)/textcaps_prompt_vis_subdir/f"{textcaps_sample_split}_epochbest.json"

    textcaps_samples_data = json.load(textcaps_vis_file_path.open())

    return textcaps_samples_data

def get_sample_textvqa(split, index, *subdirs):

    textvl_ocr_data_dict = load_data_textvl_ocr(split)
    textvqa_anno_data_dict = load_anno_textvqa(split)
    answer_output_pred_list = []
    for subdir in subdirs:
        textvqa_samples_data = load_samples_textvqa(split, subdir)
        sample_data = textvqa_samples_data[index]
        question_id = sample_data['question_id']
        answer_output_pred_list.append(sample_data['answer'])

        question_anno_data = textvqa_anno_data_dict[question_id]
        image_id = question_anno_data['image_id']
        question_input = question_anno_data['question']
        if split == 'val':
            answer_output_gt = '\t\t\t'.join([f""+answer for i, answer in enumerate(question_anno_data['answers'])])
        else:
            answer_output_gt = 'No GroundTruth for test split !!!'
        ocr_input = '\t\t\t'.join(textvl_ocr_data_dict[image_id]['ocr_tokens'])
        image_split = 'trainval' if split == 'val' else 'test'
        image_input = Path(TEXTVL_IMAGE_ROOT)/image_split/f"{image_id}.jpg"
    
    return [str(image_input), ocr_input, question_input, answer_output_gt, *answer_output_pred_list]

def get_sample_textcaps(split, index, *subdirs):

    textvl_ocr_data_dict = load_data_textvl_ocr(split)
    textcaps_anno_data_dict = load_anno_textcaps(split)
    caption_output_pred_list = []
    for subdir in subdirs:
        textcaps_samples_data = load_samples_textcaps(split, subdir)
        sample_data = textcaps_samples_data[index]
        image_id = sample_data['image_id']
        caption_output_pred_list.append(sample_data['caption'])

        caption_anno_data = textcaps_anno_data_dict[image_id]
        if split == 'val':
            caption_output_gt = '\n'.join([f""+caption for i, caption in enumerate(caption_anno_data['reference_strs']) ])
        else:
            caption_output_gt = 'No GroundTruth for test split !!!'
        ocr_input = '\t\t\t'.join(textvl_ocr_data_dict[image_id]['ocr_tokens'])
        image_split = 'trainval' if split == 'val' else 'test'
        image_input = Path(TEXTVL_IMAGE_ROOT)/image_split/f"{image_id}.jpg"

    return [str(image_input), ocr_input, caption_output_gt, *caption_output_pred_list]

def random_select_textvqa(split, *subdirs):

    textvqa_samples_data = load_samples_textvqa(split, subdirs[0])
    index = random.randint(1, len(textvqa_samples_data))
    sample = get_sample_textvqa(split, index, *subdirs)
    sample.append(index)
    return sample

def random_select_textcaps(split, *subdirs):

    textcaps_samples_data = load_samples_textcaps(split, subdirs[0])
    index = random.randint(1, len(textcaps_samples_data))
    sample = get_sample_textcaps(split, index, *subdirs)
    sample.append(index)
    return sample