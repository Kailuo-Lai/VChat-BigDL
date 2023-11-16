import os
from transformers import AutoTokenizer
from langchain.chains import ConversationalRetrievalChain, StuffDocumentsChain
from langchain.prompts import PromptTemplate
from bigdl.llm.langchain.llms import TransformersLLM
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from bigdl.llm.langchain.embeddings import TransformersEmbeddings
from langchain import LLMChain
import langid

# from models.helsinki_model import Translator
from utils.utils import new_cd

parent_dir = os.path.dirname(__file__)
langid.set_languages(["en", "zh"])

# 英文prompt
condense_template_en = """
You are ChatGLM3, an AI assistant designed for helping users. Follow the user's instructions carefully.
Assume the discussion is about the video content.
Combine the following conversation and a follow up question and use English to rephrase the follow up question to be a standalone question.
Chat History:
=========
{chat_history}
=========
Follow Up Question: {question}
Standalone question:
"""

qa_template_en = """
You are an AI assistant designed for answering questions about a video.
You are given a timeline document. The document records what people see and hear from a single video, as well as the time interval in which relevant images and audio are recorded.
Try to connet these information and provide a conversational answer in English.
=========
{context}
=========
Question: {question}
Answer: 
"""

# 中文prompt
condense_template_zh = """
你是一款专为帮助用户而设计的AI助手。仔细按照用户的指示进行操作。
假设以下讨论是关于视频内容的。
结合下面的对话和一个问题，用中文将问题重新表述为一个独立的问题。
聊天记录：
=========
{chat_history}
=========
问题：{question}
独立问题：
"""

qa_template_zh = """
你是一个专门回答有关视频的问题而设计的AI助手。
以下是一个关于某个视频的时间线文件，该文档记录了人们从单个视频中看到和听到的内容，以及记录到相关画面与音频的时间区间。
试着把这些信息联系起来，用中文回答有关问题。
=========
{context}
=========
问题：{question}
回答：
"""

CONDENSE_QUESTION_PROMPT_EN = PromptTemplate.from_template(condense_template_en)
QA_PROMPT_EN = PromptTemplate(template=qa_template_en, input_variables=["question", "context"])
DOC_PROMPT = PromptTemplate.from_template("{page_content}")

CONDENSE_QUESTION_PROMPT_ZH = PromptTemplate.from_template(condense_template_zh)
QA_PROMPT_ZH = PromptTemplate(template=qa_template_zh, input_variables=["question", "context"])

class LlmReasoner():
    def __init__(self, args):
        self.history = []
        self.llm_version = args.llm_version
        self.embed_version = args.embed_version
        self.qa_chain = None
        self.vectorstore = None
        self.top_k = args.top_k
        self.qa_max_new_tokens = args.qa_max_new_tokens
        self.init_model()
    
    def init_model(self):
        with new_cd(parent_dir):
            self.llm = TransformersLLM.from_model_id_low_bit(f"../checkpoints/{self.llm_version}",
                                                             {"trust_remote_code": True})
            self.llm.streaming = False
            self.embeddings = TransformersEmbeddings.from_model_id(model_id=f"../checkpoints/{self.embed_version}")
            self.embeddings.encode_kwargs = {"truncation": True, "max_length": 512, "padding": True}
        self.question_generator = LLMChain(llm=self.llm, prompt=CONDENSE_QUESTION_PROMPT_EN)
        self.answer_generator = LLMChain(llm=self.llm, prompt=QA_PROMPT_EN, 
                                         llm_kwargs={"max_new_tokens": self.qa_max_new_tokens})
        self.doc_chain = StuffDocumentsChain(llm_chain=self.answer_generator, document_prompt=DOC_PROMPT,
                                             document_variable_name='context')

        self.text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0, keep_separator=True)
        
    def create_qa_chain(self, en_input_log):
        en_texts = self.text_splitter.split_text(en_input_log)
        self.vectorstore_en = FAISS.from_texts(en_texts, self.embeddings, metadatas=[{"video_clip": str(i)} for i in range(len(en_texts))])
        self.retriever_en = self.vectorstore_en.as_retriever(search_kwargs={"k": self.top_k})
        
        self.qa_chain = ConversationalRetrievalChain(retriever=self.retriever_en,
                                        question_generator=self.question_generator,
                                        combine_docs_chain=self.doc_chain,
                                        return_generated_question = True,
                                        return_source_documents = True,
                                        rephrase_question=False)
    
    def __call__(self, question):
        lid = langid.classify(question)
        print(lid)
        if lid[0] == "zh":
            self.qa_chain.question_generator.prompt = CONDENSE_QUESTION_PROMPT_ZH
            self.qa_chain.combine_docs_chain.llm_chain.prompt = QA_PROMPT_ZH
        response = self.qa_chain({"question": question, "chat_history": self.history})
        self.qa_chain.question_generator.prompt = CONDENSE_QUESTION_PROMPT_EN
        self.qa_chain.combine_docs_chain.llm_chain.prompt = QA_PROMPT_EN
        try:
            answer = response["answer"].split("Answer: \n")[1] if "Answer: \n" in response["answer"] else response["answer"].split("回答：\n")[1]
        except:
            pass
        generated_question = response["generated_question"]
        source_documents = response["source_documents"]
        self.history.append([question, answer])
        return self.history, generated_question, source_documents, lid[0]
    
    def clean_history(self):
        self.history = []