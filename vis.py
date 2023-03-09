import json
import argparse
import gradio

def parse_args():
    parser = argparse.ArgumentParser(
        prog='TextVQA and TextCaps Visualization Gradio Demo',
        description='This demo can visualize image&caption or image&question&answer in one single page.',
        epilog='Text at the bottom of help')

    parser.add_argument('-d', 'res_dir', type=str, default="./textvqa_zeroshot_with_rosetta_flant5xl_eval", help="visualization result directory")
    parser.add_argument('-s', '--split', type=str, default="val", choices=["val", "test"], help="val or test split result")
    parser.add_argument('-t', '--task', type=str, default="textvqa", choices=["textvqa", "textcaps"], help="which task to visualize")
    parser.add_argument('-r', '--random', action='store_true', help="random select or next select")

    return parser.parse_args()


def load_data(args):
    res_dir = args.res_dir
    split = args.split
    task = args.task
    is_random = args.random

    if task == 'textvqa':
        filename = f"{split}_vqa_result.json"
    else:
        filename = f"{split}_epochbest.json"

    data = json.load(open(f"{res_dir}/{filename}", 'r'))


def visualize_textcaps(index):


def visualize_textvqa(index):


def run_gradio_demo(args, data):
    if args.task == 'textvqa':
        title = "TextVQA Result Visualization"
        inputs = [
            gr.Slider(minimum=0, maximum=len(data), step=1, value=1, label="Result Index"),
        ]
        outputs = [
            gr.Image()
            gr.Textbox(label="Caption")
        ]
        demo = gr.Interface(
            visualize_textvqa,
            inputs=inputs,
            outputs=outputs,
            title=title,
        )
    else:
        title = "TextCaps Result Visualization"
        inputs = [
            gr.Slider(minimum=0, maximum=len(data), step=1, value=1, label="Result Index"),
        ]
        outputs = [
            gr.Textbox(label="Question")
            gr.Textbox(label="Answer")
        ]
        demo = gr.Interface(
            visualize_textcaps,
            inputs=inputs,
            outputs=outputs,
            title=title,
        )
    demo.launch(share=True)

def main(args):
    data = load_data(args)
    run_gradio_demo(args, data)

if __name__ == '__main__':
    args = parse_args()
    main(args)
