# -*- coding: utf-8 -*-
'''
@File    :   main_gradio.py
@Time    :   2023/10/18 13:06:26
@Author  :   LCR
'''

import argparse
import gradio as gr
import os
import shutil

from models.vchat_bigdl import VChat
from utils.bilibili_video_downloader import download_bilibili_video

parser = argparse.ArgumentParser()

# kts arguments
parser.add_argument("--alpha", default=10, type=int, help="Determine the maximum segment number for KTS algorithm, the larger the value, the fewer segments.")
parser.add_argument("--vmax", default=1, type=float, help="Special parameter of penalty term for KTS algorithm, the larger the value, the fewer segments.")
parser.add_argument("--beta", default=1, type=int, help="The smallest time gap between successive clips, in seconds.")

# clip model arguments
parser.add_argument("--clip_version", default="clip-vit-base-patch32", help="Clip model version for video feature extractor")

# tag2text model arguments
parser.add_argument("--tag2text_thershld", default=0.68, type=float, help="Threshold for tag2text model")

# whisper model arguments
parser.add_argument("--whisper_version", default="medium", help="Whisper model version for video asr")

# llm model arguments
parser.add_argument("--llm_version", default="Llama-2-7b-chat-hf-INT4", help="LLM model version")
parser.add_argument("--embed_version", default="all-MiniLM-L12-v2", help="Embedding model version")
parser.add_argument("--top_k", default=3, type=int, help="Return top k relevant contexts to llm")
parser.add_argument("--qa_max_new_tokens", default=128, type=int, help="Number of max new tokens for llm")

# general arguments
parser.add_argument("--port", type = int, default = 8899, help = "Gradio server port")
parser.add_argument("--lid", default="en", choices=['en', 'zh'], help="which language do you want use during conversation")

args = parser.parse_args()
print(args)

vchat = VChat(args)
vchat.init_model()

global_chat_history = []
global_lid = args.lid
global_en_log_result = ""
global_zh_log_result = ""

def clean_conversation():
    global global_chat_history
    vchat.clean_history()
    global_chat_history = []
    shutil.rmtree('./temp/bilibili_video')  
    os.mkdir('./temp/bilibili_video')
    return '', gr.update(value=None, interactive=True), None, gr.update(value=None, visible=True)

def clean_chat_history():
    global global_chat_history
    vchat.clean_history()
    global_chat_history = []
    return '', None

def submit_message(message, lid):
    chat_history, generated_question, source_documents = vchat.chat2video(message, lid)
    global_chat_history.append((message, chat_history[0][1]))
    source_documents = "".join([x.page_content for x in source_documents])
    return '', global_chat_history
    

def log_fn(vid_path, lid):
    print(vid_path)
    global global_en_log_result
    global global_zh_log_result
    if vid_path is None:
        log_text = "====== Please upload video or provide bilibili_BVid 🙃====="
        gr.update(value=log_text, visible=True)
    else:
        global_en_log_result, global_zh_log_result = vchat.video2log(vid_path)
        if lid == "en":
            return gr.update(value=global_en_log_result, visible=True)
        elif lid == "zh":
            return   gr.update(value=global_zh_log_result, visible=True)

def vmax_change(vmax):
    vchat.video_segmenter.vmax = vmax
    print(f"\033[35;1mvmax={vchat.video_segmenter.vmax}" + '\033[0m')
    
def lid_change(lid):
    global global_chat_history
    vchat.clean_history()
    global_chat_history = []
    print(f"\033[31;1mChange to {lid}\033[0m")
    if lid == "en":
        global global_en_log_result
        return gr.update(value=global_en_log_result, visible=True), '', None
    elif lid == "zh":
        global global_zh_log_result
        return gr.update(value=global_zh_log_result, visible=True), '', None

def subvid_fn(bvid):
    print(bvid)
    shutil.rmtree('./temp/bilibili_video')  
    os.mkdir('./temp/bilibili_video')
    save_path = download_bilibili_video(bvid)
    return gr.update(value=save_path), '', None, gr.update(value=None, visible=True)

css = """
      #col-container {max-width: 80%; margin-left: auto; margin-right: auto;}
      #video_inp {min-height: 100px}
      #chatbox {min-height: 100px;}
      #header {text-align: center;}
      #hint {font-size: 1.0em; padding: 0.5em; margin: 0;}
      .message { font-size: 1.2em; }
      """

with gr.Blocks(css=css) as demo:


    with gr.Column(elem_id="col-container"):
        gr.Markdown("""## 🤖VChat BigDL
                    Powered by BigDL, Llama, Clip, Whisper, Tag2Text, Helsinki and LangChain
                    Inspired by showlab/log""",
                    elem_id="header")
        lid_choice = gr.Dropdown(choices=["en", "zh"], value="en", label="Choose language")

        with gr.Row():
            with gr.Column():
                video_inp = gr.Video(label="video_input")
                gr.Markdown("Input bilibili BV id in this textbox, *e.g.* *BV1mr4y1f7hr*", elem_id="hint")
                with gr.Row():
                    video_id = gr.Textbox(value="", placeholder="bilibili BV", show_label=False)
                    vidsub_btn = gr.Button("Submit bilibili Video")
                
                chatbot = gr.Chatbot(elem_id="chatbox")
                input_message = gr.Textbox(show_label=False, placeholder="Enter text and press enter", visible=True).style(container=False)
                btn_submit = gr.Button("Submit")
                with gr.Row():
                    btn_clean_chat_history = gr.Button("Clean Chat History")
                    btn_clean_conversation = gr.Button("🔃 Start New Conversation")
                    
            
            with gr.Column():
                vmax_slider = gr.Slider(minimum=0, maximum=2, value=args.vmax, label='vmax',
                                        info='Special parameter of penalty term for KTS algorithm, the larger the value, the fewer video clips.')
                log_btn = gr.Button("Generate Video Document")
                log_outp = gr.Textbox(label="Document output\nPlease be patient", lines=40)
                total_tokens_str = gr.Markdown(elem_id="total_tokens_str")

    lid_choice.change(lid_change, [lid_choice], [log_outp, input_message, chatbot])
    btn_submit.click(submit_message, [input_message, lid_choice], [input_message, chatbot])
    input_message.submit(submit_message, [input_message, lid_choice], [input_message, chatbot])
    btn_clean_conversation.click(clean_conversation, [], [input_message, video_inp, chatbot, log_outp])
    btn_clean_chat_history.click(clean_chat_history, [], [input_message, chatbot])
    log_btn.click(log_fn, [video_inp, lid_choice], [log_outp])
    vmax_slider.release(vmax_change, [vmax_slider], [])
    vidsub_btn.click(subvid_fn, [video_id], [video_inp, input_message, chatbot, log_outp])

    demo.load(queur=False)


demo.queue(concurrency_count=10)
demo.launch(height='800px', server_port=args.port, debug=True, share=True)