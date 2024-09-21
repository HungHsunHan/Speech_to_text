import ctypes
import os
import subprocess
import tempfile

import gradio as gr
import whisper
import yt_dlp
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# Try to load the OpenMP library
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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


def convert_audio(input_path, output_path):
    command = [
        "ffmpeg",
        "-i",
        input_path,
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        output_path,
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting audio: {e.stderr}")
        raise


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


def process_video(youtube_url, prompt, model_type, chunk_size):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "video")

            print("Downloading and extracting audio...")
            download_youtube_video(youtube_url, video_path)

            print(f"Transcribing audio using {model_type} model...")
            transcription = transcribe_audio(video_path + ".wav", model_type)

            print(f"Summarizing the transcription with chunk size {chunk_size}...")
            summary = summarize_text(transcription, prompt, chunk_size)

            print("Process completed successfully.")
            return summary

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        return error_message


def download_file(file_path):
    if os.path.exists(file_path):
        return file_path
    else:
        return "File not found. Please try processing the video again."


# Gradio interface
iface = gr.Interface(
    fn=process_video,
    inputs=[
        gr.Textbox(label="YouTube URL"),
        gr.Textbox(
            "Summarize the main points of this video",
            label="Summarization Prompt",
        ),
        gr.Dropdown(
            ["base", "small", "medium"], value="base", label="Whisper Model Type"
        ),
        gr.Number(value=1000, label="Chunk Size", precision=0),
    ],
    outputs=[
        gr.Textbox(label="Summary"),
    ],
    title="Video Summarizer",
    description="Enter a YouTube URL, a prompt to summarize the video content, choose the Whisper model type, and set the chunk size for text splitting.",
)

if __name__ == "__main__":
    iface.launch()
