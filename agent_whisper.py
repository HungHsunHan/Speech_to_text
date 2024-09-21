import json
import os
import tempfile
from typing import Dict, List, Tuple

import gradio as gr
import whisper
import yt_dlp
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import Tool
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydub import AudioSegment

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
    azure_deployment="gpt-4o",
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

# Agent functions


def extract_video_info(question: str) -> str:
    """Extract video link, time range, and model type from the user's question."""
    prompt = PromptTemplate(
        input_variables=["question"],
        template="""Extract the YouTube video link, time range (in minutes), and preferred Whisper model type (base, small, or medium) from the following question: {question}
        Provide the results in a JSON format with keys 'video_link', 'start_time', 'end_time', and 'model_type'.
        If the time range is not specified, use 0 for start_time and -1 for end_time to indicate the entire video.
        If the model type is not specified, use 'base' as the default.""",
    )

    result = llm.predict(prompt.format(question=question))

    try:
        extracted_info = json.loads(result)
        # Ensure all required keys are present
        required_keys = ["video_link", "start_time", "end_time", "model_type"]
        for key in required_keys:
            if key not in extracted_info:
                extracted_info[key] = (
                    0
                    if key == "start_time"
                    else (
                        -1
                        if key == "end_time"
                        else "base" if key == "model_type" else ""
                    )
                )
        return json.dumps(extracted_info)
    except json.JSONDecodeError:
        # If JSON parsing fails, return a default JSON string
        default_info = {
            "video_link": (
                question if "youtube.com" in question or "youtu.be" in question else ""
            ),
            "start_time": 0,
            "end_time": -1,
            "model_type": "base",
        }
        return json.dumps(default_info)


def download_youtube_video(url: str, output_path: str) -> None:
    try:
        """Download the audio from a YouTube video."""
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
    except Exception as e:
        print("download")
        print(e)
        return f"download: {e}"


def extract_audio_segment(
    input_path: str, output_path: str, start_time: float, end_time: float
) -> None:
    """Extract a segment of audio from the full audio file."""
    audio = AudioSegment.from_wav(input_path)
    start_ms = start_time * 60 * 1000
    end_ms = end_time * 60 * 1000 if end_time > 0 else len(audio)
    segment = audio[start_ms:end_ms]
    segment.export(output_path, format="wav")


def transcribe_audio(audio_path: str, model_type: str) -> str:
    """Transcribe the audio using the specified Whisper model."""
    model = whisper.load_model(model_type)
    result = model.transcribe(audio_path)
    return result["text"]


def summarize_text(text: str, prompt: str, chunk_size: int) -> str:
    """Summarize the transcribed text using the specified prompt and chunk size."""
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    texts = text_splitter.split_text(text)
    docsearch = FAISS.from_texts(texts, embeddings)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
    )
    summary = qa_chain.run(prompt)
    return summary


# Agent definitions


class VideoProcessingAgent:
    def __init__(self):

        self.tools = [
            Tool(
                name="Extract Video Info",
                func=extract_video_info,
                description="Extracts video link, time range, and model type from the user's question",
            ),
            Tool(
                name="Download YouTube Video",
                func=lambda x: download_youtube_video(x["url"], x["output_path"]),
                description="Downloads the audio from a YouTube video. Input should be a dictionary with 'url' and 'output_path' keys.",
            ),
            Tool(
                name="Extract Audio Segment",
                func=lambda x: extract_audio_segment(
                    x["input_path"], x["output_path"], x["start_time"], x["end_time"]
                ),
                description="Extracts a segment of audio from the full audio file. Input should be a dictionary with 'input_path', 'output_path', 'start_time', and 'end_time' keys.",
            ),
            Tool(
                name="Transcribe Audio",
                func=lambda x: transcribe_audio(x["audio_path"], x["model_type"]),
                description="Transcribes the audio using the specified Whisper model. Input should be a dictionary with 'audio_path' and 'model_type' keys.",
            ),
            Tool(
                name="Summarize Text",
                func=lambda x: summarize_text(x["text"], x["prompt"], x["chunk_size"]),
                description="Summarizes the transcribed text using the specified prompt and chunk size. Input should be a dictionary with 'text', 'prompt', and 'chunk_size' keys.",
            ),
        ]
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.agent = initialize_agent(
            self.tools,
            llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
        )

    def process_video(self, question: str) -> str:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                full_audio_path = os.path.join(temp_dir, "video.wav")
                segment_audio_path = os.path.join(temp_dir, "segment_video.wav")

                # Extract video information
                video_info = self.agent.run(
                    f"Extract video information from this question: {question}"
                )
                video_info = eval(
                    video_info
                )  # Convert string representation to dictionary

                # Download video
                self.agent.run(
                    f"Download the YouTube video from {video_info['video_link']} to the output path which is {full_audio_path}\
                    Use {full_audio_path} as the output_path of the fucntion download_youtube_video"
                )

                # Extract audio segment
                self.agent.run(
                    f"Extract audio segment from {full_audio_path} to {segment_audio_path} "
                    f"with start time {video_info['start_time']} and end time {video_info['end_time']}"
                )

                # Transcribe audio
                transcription = self.agent.run(
                    f"Transcribe the audio at {segment_audio_path} using the {video_info['model_type']} model"
                )

                # Summarize text
                summary = self.agent.run(
                    f"Summarize the following transcription with the prompt '{question}' and chunk size 1000: {transcription}"
                )

                return summary

        except Exception as e:
            return f"An error occurred: {str(e)}"


# Gradio interface
def process_video_question(question: str) -> str:
    agent = VideoProcessingAgent()
    return agent.process_video(question)


iface = gr.Interface(
    fn=process_video_question,
    inputs=[
        gr.Textbox(
            label="Ask a question about a YouTube video (include the video link in your question)"
        ),
    ],
    outputs=[
        gr.Textbox(label="Summary"),
    ],
    title="Video Content Summarizer",
    description="Ask a question about a YouTube video, and the system will extract the relevant information, transcribe the video, and provide a summary based on your question.",
)

if __name__ == "__main__":
    iface.launch()
