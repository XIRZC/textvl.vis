import platform

# CONSTANT DEFINITION
OS = platform.system() # Windows or Linux
VIS_ROOT = '.'
TEXTVQA_VIS_ROOT = f"{VIS_ROOT}/textvqa"
TEXTCAPS_VIS_ROOT = f"{VIS_ROOT}/textcaps"
TEXTVL_IMAGE_ROOT = f"{VIS_ROOT}/TextVL/Images"
TEXTVQA_ANNOTATION_ROOT = f"{VIS_ROOT}/TextVL/Annotations/VQA"
TEXTCAPS_ANNOTATION_ROOT = f"{VIS_ROOT}/TextVL/Annotations/IC"
TEXTVL_OCR_ANNOTATION_ROOT = f"{VIS_ROOT}/TextVL/Annotations/OCR"
TEXTVL_OCR_VERSION_DICT = {
    "Rosetta": "Rosetta_OCR_v0.2",
    "Microsoft": "Microsoft_OCR_v1.0",
    "Amazon": "Amazon_OCR_v1.0",
}
CSS_FORMAT = """
#textvqa_anno_box {background-color: rgba(37, 190, 45, 0.3)}
#textvqa_contrast_box1 {background-color: rgba(37, 131, 190, 0.3)}
#textvqa_contrast_box2 {background-color: rgba(255, 226, 63, 0.3)}
#textvqa_contrast_box3 {background-color: rgba(190, 37, 56, 0.3)}
#textcaps_anno_box {background-color: rgba(37, 190, 45, 0.3)}
#textcaps_contrast_box1 {background-color: rgba(37, 131, 190, 0.3)}
#textcaps_contrast_box2 {background-color: rgba(255, 226, 63, 0.3)}
#textcaps_contrast_box3 {background-color: rgba(190, 37, 56, 0.3)}
"""
