# CONSTANT DEFINITION
NUM_CONTRAST = 2 # WithOCR, WithoutOCR, M4C/TAP
VIS_ROOT = '/home/mrxir/Codes/textvl.vis'
TEXTVQA_VIS_ROOT = f"{VIS_ROOT}/textvqa"
TEXTCAPS_VIS_ROOT = f"{VIS_ROOT}/textcaps"
TEXTVL_IMAGE_ROOT = f"{VIS_ROOT}/TextVL/Images"
TEXTVQA_ANNOTATION_ROOT = f"{VIS_ROOT}/TextVL/Annotations/VQA"
TEXTCAPS_ANNOTATION_ROOT = f"{VIS_ROOT}/TextVL/Annotations/IC"
TEXTVL_OCR_ANNOTATION_ROOT = f"{VIS_ROOT}/TextVL/Annotations/OCR"
CSS_FORMAT = """
#textvqa_anno_box {background-color: rgba(37, 190, 45, 0.3)}
#textvqa_contrast_box1 {background-color: rgba(37, 131, 190, 0.3)}
#textvqa_contrast_box2 {background-color: rgba(255, 226, 63, 0.3)}
#textvqa_contrast_box3 {background-color: rgba(190, 37, 56, 0.3)}
#textcaps_anno_box {background-color: rgba(37, 190, 45, 0.3)}
#textcaps_contrast_box1 {background-color: rgba(37, 131, 190, 0.3)}
#textcaps_contrast_box2 {background-color: rgba(255, 226, 63, 0.3)}
#textcaps_contrast_box3 {background-color: rgba(190, 37, 56, 0.3}
"""