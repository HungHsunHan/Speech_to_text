import os
import tempfile
from typing import Dict, List, Tuple

import gradio as gr
import whisper
import yt_dlp
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from llama_index.core.agent import AgentRunner, FunctionCallingAgentWorker, ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from pydub import AudioSegment

# Load environment variables
load_dotenv()


llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)


def extract_video_info(question: str) -> Dict[str, str]:
    """Extract video link, time range, and model type from the user's question."""
    prompt = PromptTemplate(
        input_variables=["question"],
        template="""Extract the YouTube video link, time range (in minutes), and preferred Whisper model type (base, small, or medium) from the following question: {question}
        Provide the results in a dictionary format with keys 'video_link', 'start_time', 'end_time', and 'model_type'.
        If the time range is not specified, use 0 for start_time and -1 for end_time to indicate the entire video.
        If the model type is not specified, use 'base' as the default.""",
    )

    result = llm.predict(prompt.format(question=question))

    try:
        extracted_info = eval(result)
        # Ensure all required keys are present
        required_keys = ["video_link", "start_time", "end_time", "model_type"]
        for key in required_keys:
            if key not in extracted_info:
                extracted_info[key] = (
                    "0"
                    if key == "start_time"
                    else (
                        "-1"
                        if key == "end_time"
                        else "base" if key == "model_type" else ""
                    )
                )
        return extracted_info
    except:
        # If eval fails or doesn't return a dictionary, return a default dictionary
        return {
            "video_link": (
                question if "youtube.com" in question or "youtu.be" in question else ""
            ),
            "start_time": "0",
            "end_time": "-1",
            "model_type": "base",
        }


def download_youtube_video(url: str, output_path: str) -> None:
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


# Updated VideoProcessingAgent
class VideoProcessingAgent:
    def __init__(self):
        self.tools = [
            FunctionTool.from_defaults(
                fn=extract_video_info,
                name="Extract Video Info",
                description="Extracts video link, time range, and model type from the user's question",
            ),
            FunctionTool.from_defaults(
                fn=download_youtube_video,
                name="Download YouTube Video",
                description="Downloads the audio from a YouTube video and save it into the defined output path",
            ),
            FunctionTool.from_defaults(
                fn=extract_audio_segment,
                name="Extract Audio Segment",
                description="Extracts a segment of audio from the full audio file",
            ),
            FunctionTool.from_defaults(
                fn=transcribe_audio,
                name="Transcribe Audio",
                description="Transcribes the audio using the specified Whisper model",
            ),
            FunctionTool.from_defaults(
                fn=summarize_text,
                name="Summarize Text",
                description="Summarizes the transcribed text using the specified prompt and chunk size",
            ),
        ]

        self.agent = FunctionCallingAgentWorker.from_tools(
            tools=self.tools,
            llm=llm,
            verbose=True,
        )

        self.agent_runner = AgentRunner(self.agent)

    def process_video(self, question: str) -> str:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                full_audio_path = os.path.join(temp_dir, "video.wav")
                segment_audio_path = os.path.join(temp_dir, "segment_video.wav")

                # Extract video information
                # video_info = self.agent_runner.chat(
                #     f"Extract video information from this question: {question}",
                #     tool_name="Extract Video Info",
                # )

                # # Download video
                # self.agent_runner.run(
                #     f"Download the YouTube video from {video_info['video_link']} to {full_audio_path}",
                #     tool_name="Download YouTube Video",
                # )

                # # Extract audio segment
                # self.agent_runner.run(
                #     f"Extract audio segment from {full_audio_path} to {segment_audio_path} with start time {video_info['start_time']} and end time {video_info['end_time']}",
                #     tool_name="Extract Audio Segment",
                # )

                # # Transcribe audio
                # transcription = self.agent_runner.run(
                #     f"Transcribe the audio at {segment_audio_path} using the {video_info['model_type']} model",
                #     tool_name="Transcribe Audio",
                # )

                # # Summarize text
                # summary = self.agent_runner.run(
                #     f"Summarize the following transcription with the prompt '{question}' and chunk size 1000: {transcription}",
                #     tool_name="Summarize Text",
                # )
                response = self.agent_runner.chat(question)

                return response

        except Exception as e:
            return f"An error occurred: {str(e)}"


# Gradio interface (unchanged)
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
