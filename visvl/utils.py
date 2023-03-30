from pathlib import Path
import json
import random
from .consts import *

def get_file_list(root_path):
    separator = '\\' if OS == 'Windows' else '/'
    file_list = sorted(list(set([separator.join(str(item).split(separator)[-3:-1]) for item in Path(root_path).resolve().glob('**/*') if item.is_file()])))
    return file_list

def dict_list2dict_dict(dict_list, key_name):
    dict_dict = dict()
    for item in dict_list:
        if item[key_name] not in dict_dict:
            dict_dict[item[key_name]] = dict()
        dict_dict[item[key_name]] = item
    return dict_dict

def load_anno_textvl_ocr(split, format="list"):

    textvl_ocr_file_path = Path(TEXTVL_OCR_ANNOTATION_ROOT).resolve()/f"TextVQA_Rosetta_OCR_v0.2_{split}.json"

    textvl_ocr_data = json.load(textvl_ocr_file_path.open())['data']
    if format == "dict":
        return dict_list2dict_dict(textvl_ocr_data, 'image_id')
    elif format == "list":
        return textvl_ocr_data
    else: # "both" for first list second dict
        return textvl_ocr_data, dict_list2dict_dict(textvl_ocr_data, 'image_id')
    

def load_anno_textvqa(textvqa_sample_split, format="list"):

    textvqa_anno_file_path = Path(TEXTVQA_ANNOTATION_ROOT).resolve()/f"TextVQA_0.5.1_{textvqa_sample_split}.json"

    textvqa_anno_data = json.load(textvqa_anno_file_path.open())['data']
    if format == "dict":
        return dict_list2dict_dict(textvqa_anno_data, 'question_id')
    elif format == "list":
        return textvqa_anno_data
    else: # "both" for first list second dict
        return textvqa_anno_data, dict_list2dict_dict(textvqa_anno_data, 'question_id')

def load_anno_textcaps(textcaps_sample_split, format="list"):

    textcaps_anno_file_path = Path(TEXTCAPS_ANNOTATION_ROOT).resolve()/f"TextCaps_0.1_{textcaps_sample_split}.json"

    textcaps_anno_data = json.load(textcaps_anno_file_path.open())['data']
    if format == "dict":
        return dict_list2dict_dict(textcaps_anno_data, 'image_id')
    elif format == "list":
        return textcaps_anno_data
    else: # "both" for first list second dict
        return textcaps_anno_data, dict_list2dict_dict(textcaps_anno_data, 'image_id')


textcaps_anno_list = dict()
textvqa_anno_list = dict()
textvl_ocr_anno_list = dict()
textcaps_anno_dict = dict()
textvqa_anno_dict = dict()
textvl_ocr_anno_dict = dict()
for split in ['val', 'test']:
    textvqa_anno_list[split], textvqa_anno_dict[split] = load_anno_textvqa(split, format="both")
    textcaps_anno_list[split], textcaps_anno_dict[split] = load_anno_textcaps(split, format="both")
    textvl_ocr_anno_list[split], textvl_ocr_anno_dict[split] = load_anno_textvl_ocr(split, format="both")


def load_samples_textvqa(textvqa_sample_split, textvqa_prompt_vis_subdir, format="dict"):

    textvqa_vis_file_path = Path(TEXTVQA_VIS_ROOT).resolve()/textvqa_prompt_vis_subdir/f"{textvqa_sample_split}_vqa_result.json"

    textvqa_samples_data = json.load(textvqa_vis_file_path.open())
    if format == "dict":
        return dict_list2dict_dict(textvqa_samples_data, 'question_id')
    else:
        return textvqa_samples_data

def load_samples_textcaps(textcaps_sample_split, textcaps_prompt_vis_subdir, format="dict"):

    textcaps_vis_file_path = Path(TEXTCAPS_VIS_ROOT).resolve()/textcaps_prompt_vis_subdir/f"{textcaps_sample_split}_epochbest.json"

    textcaps_samples_data = json.load(textcaps_vis_file_path.open())
    if format == "dict":
        return dict_list2dict_dict(textcaps_samples_data, 'image_id')
    else:
        return textcaps_samples_data

def get_sample_textvqa(split, index, *subdirs):

    textvqa_anno_data_list = textvqa_anno_list[split]
    textvl_ocr_data_dict = textvl_ocr_anno_dict[split]
    answer_output_pred_list = []
    for subdir in subdirs:
        textvqa_samples_data_dict = load_samples_textvqa(split, subdir, format="dict")
        question_anno_data = textvqa_anno_data_list[index]
        question_id = question_anno_data['question_id']

        sample_data = textvqa_samples_data_dict[question_id]
        answer_output_pred_list.append(sample_data['answer'])

        image_id = question_anno_data['image_id']
        question_input = question_anno_data['question']
        if split == 'val':
            answer_output_gt = '\t\t\t'.join([f""+answer for i, answer in enumerate(question_anno_data['answers'])])
        else:
            answer_output_gt = 'No GroundTruth for test split !!!'
        ocr_input = '\t\t\t'.join(textvl_ocr_data_dict[image_id]['ocr_tokens'])
        image_split = 'trainval' if split == 'val' else 'test'
        image_input = Path(TEXTVL_IMAGE_ROOT).resolve()/image_split/f"{image_id}.jpg"
    
    return [str(image_input), ocr_input, question_input, answer_output_gt, *answer_output_pred_list]

def get_sample_textcaps(split, index, *subdirs):

    textcaps_anno_data_list = textcaps_anno_list[split]
    textvl_ocr_data_dict = textvl_ocr_anno_dict[split]
    caption_output_pred_list = []
    for subdir in subdirs:
        textcaps_samples_data_dict = load_samples_textcaps(split, subdir, format="dict")
        caption_anno_data = textcaps_anno_data_list[index]
        image_id = caption_anno_data['image_id']

        sample_data = textcaps_samples_data_dict[image_id]
        caption_output_pred_list.append(sample_data['caption'])

        if split == 'val':
            caption_output_gt = '\n'.join([f""+caption for i, caption in enumerate(caption_anno_data['reference_strs']) ])
        else:
            caption_output_gt = 'No GroundTruth for test split !!!'
        ocr_input = '\t\t\t'.join(textvl_ocr_data_dict[image_id]['ocr_tokens'])
        image_split = 'trainval' if split == 'val' else 'test'
        image_input = Path(TEXTVL_IMAGE_ROOT).resolve()/image_split/f"{image_id}.jpg"

    return [str(image_input), ocr_input, caption_output_gt, *caption_output_pred_list]

def random_select_textvqa(split, *subdirs):

    # textvqa_samples_data = load_samples_textvqa(split, subdirs[0])
    textvqa_anno_data = textvqa_anno_list[split]
    index = random.randint(1, len(textvqa_anno_data))
    sample = get_sample_textvqa(split, index, *subdirs)
    sample.append(index)
    return sample

def random_select_textcaps(split, *subdirs):

    # textcaps_samples_data = load_samples_textcaps(split, subdirs[0])
    textcaps_anno_data = textcaps_anno_list[split]
    index = random.randint(1, len(textcaps_anno_data))
    sample = get_sample_textcaps(split, index, *subdirs)
    sample.append(index)
    return sample