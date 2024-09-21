import ctypes
import os
import subprocess
import tempfile
from pathlib import Path

import gradio as gr
import whisper
import yt_dlp
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydub import AudioSegment

# Set the KMP_DUPLICATE_LIB_OK environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Try to load the OpenMP library
try:
    ctypes.CDLL("libomp.dylib")
except OSError:
    pass

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-small",
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

# Initialize Azure OpenAI language model
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",  # or your specific deployment name
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)


def download_youtube_video(url, output_path):
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "outtmpl": output_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def extract_audio_segment(input_path, output_path, start_time, end_time):
    audio = AudioSegment.from_wav(input_path)
    start_ms = start_time * 60 * 1000  # convert minutes to milliseconds
    end_ms = end_time * 60 * 1000 if end_time > 0 else len(audio)
    segment = audio[start_ms:end_ms]
    segment.export(output_path, format="wav")


def transcribe_audio(audio_path, model_type):
    model = whisper.load_model(model_type)
    try:
        result = model.transcribe(audio_path)
        return result["text"]
    except RuntimeError as e:
        print(f"Error transcribing audio: {str(e)}")
        raise


def summarize_text(text, prompt, chunk_size):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    texts = text_splitter.split_text(text)
    docsearch = FAISS.from_texts(texts, embeddings)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
    )
    summary = qa_chain.run(prompt)
    return summary


def process_video(youtube_url, prompt, model_type, chunk_size, start_time, end_time):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            full_audio_path = os.path.join(temp_dir, "video")
            segment_audio_path = os.path.join(temp_dir, "segment_video")

            print("Downloading and extracting audio...")
            download_youtube_video(youtube_url, full_audio_path)

            print(
                f"Extracting audio segment from {start_time} to {end_time} minutes..."
            )
            extract_audio_segment(
                full_audio_path + ".wav",
                segment_audio_path + ".wav",
                start_time,
                end_time,
            )

            print(f"Transcribing audio segment using {model_type} model...")
            transcription = transcribe_audio(segment_audio_path + ".wav", model_type)

            print(f"Summarizing the transcription with chunk size {chunk_size}...")
            summary = summarize_text(transcription, prompt, chunk_size)

            print("Process completed successfully.")
            return summary

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        return error_message


# Gradio interface
iface = gr.Interface(
    fn=process_video,
    inputs=[
        gr.Textbox(label="YouTube URL"),
        gr.Textbox(
            """你是一位專業的股市分析師，負責將影片內容整理成與股市相關的報導。你的任務是將影片轉換為條理清晰、具邏輯性的摘要和總結，並儘可能真實還原內容。具體要求如下：

            1.條列式結構：分段整理影片內容，逐一說明每個段落的重要細節與重點。
            2.市場趨勢：當影片中提到市場趨勢或預測時，務必精確引用講者的原話，不要進行模糊或概略性的描述。例如，避免將內容簡化為「預測降息可能性」或「提到設備股和營建股的表現」，而應具體說明講者提到的影響與趨勢。
            3.股市焦點：僅整理與股市相關的內容，股市以外的部分可略過不做總結。
            4.Markdown格式：將最終的重點整理以Markdown格式呈現，結構清晰、層次分明，便於閱讀和理解。""",
            label="Summarization Prompt",
        ),
        gr.Dropdown(
            ["base", "small", "medium"], value="base", label="Whisper Model Type"
        ),
        gr.Number(value=1000, label="Chunk Size", precision=0),
        gr.Number(value=0, label="Start Time (minutes)", precision=1),
        gr.Number(
            value=0, label="End Time (minutes, 0 for full duration)", precision=1
        ),
    ],
    outputs=[
        gr.Textbox(label="Summary"),
    ],
    title="Video Segment Summarizer",
    description="Enter a YouTube URL, a prompt to summarize the video content, choose the Whisper model type, set the chunk size for text splitting, and specify the time range to summarize.",
)

if __name__ == "__main__":
    iface.launch()
