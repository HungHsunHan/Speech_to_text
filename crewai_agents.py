import os

import openai
import whisper
from crewai import Agent, Crew, Process, Task
from crewai_tools import tool
from pytube import YouTube

# Set the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


# Custom tool to wrap the video download function
@tool
def download_video_tool(inputs):
    """
    Download a YouTube video from the provided URL and save it to the specified directory.

    Args:
        inputs (dict): A dictionary containing the 'url' key with the YouTube video URL.

    Returns:
        str: The path to the downloaded video file.
    """
    print(f"download_video_tool [inputs]:{inputs}")
    url = inputs["url"]
    output_dir = "downloads"
    yt = YouTube(url)
    stream = yt.streams.filter(file_extension="mp4").first()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_path = stream.download(output_dir)
    print(f"download_video_tool [output]:{video_path}")
    return video_path


# Custom tool to wrap the video transcription function
@tool
def transcribe_video_tool(inputs):
    """
    Transcribe a video file into text using the Whisper model and save the transcription as a text file.

    Args:
        inputs (dict): A dictionary containing the 'video_path' key with the path to the video file.

    Returns:
        str: The path to the text file containing the transcription.
    """
    print(f"transcribe_video_tool [inputs]:{inputs}")
    video_path = inputs["video_path"]
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    text_file_path = video_path.replace(".mp4", ".txt")
    with open(text_file_path, "w", encoding="utf-8") as f:
        f.write(result["text"])

    print(f"transcribe_video_tool [output]:{text_file_path}")
    return text_file_path


# Custom tool to wrap the text summarization function
@tool
def summarize_text_tool(inputs):
    """
    Summarize the text from a given file using OpenAI's GPT model and save the summary to a new text file.

    Args:
        inputs (dict): A dictionary containing the 'text_file_path' key with the path to the text file.

    Returns:
        str: The path to the text file containing the summary.
    """
    print(f"summarize_text_tool [inputs]:{inputs}")
    text_file_path = inputs["text_file_path"]
    with open(text_file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Call the OpenAI API for summarization
    response = openai.Completion.create(
        model="gpt-4-turbo-preview",
        prompt=f"Summarize the following text:\n\n{text}",
        max_tokens=150,
        temperature=0.5,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    summary = response.choices[0].text.strip()

    # Save the summary to a new text file
    summary_file_path = text_file_path.replace(".txt", "_summary.txt")
    with open(summary_file_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"summarize_text_tool [output]:{summary_file_path}")
    return summary_file_path


# Define the agents with their respective tools
download_agent = Agent(
    role="Video Downloader",
    goal="Download a YouTube video from the provided URL.",
    verbose=True,
    backstory="You are adept at retrieving online video content efficiently.",
    tools=[download_video_tool],  # Corrected to a list
)

transcribe_agent = Agent(
    role="Video Transcriber",
    goal="Transcribe the downloaded video into a text file.",
    verbose=True,
    backstory="You have an ear for detail and can accurately convert speech to text.",
    tools=[transcribe_video_tool],  # Corrected to a list
)

summarize_agent = Agent(
    role="Text Summarizer",
    goal="Summarize the transcribed text into a concise summary.",
    verbose=True,
    backstory="You excel at distilling complex information into digestible summaries.",
    tools=[summarize_text_tool],  # Corrected to a list
)

# Define the tasks
download_task = Task(
    description="Download the YouTube video from the provided URL.",
    expected_output="The video file should be downloaded to the specified directory.",
    agent=download_agent,
)

transcribe_task = Task(
    description="Transcribe the downloaded video into a text file.",
    expected_output="The transcription should be saved as a text file.",
    agent=transcribe_agent,
)

summarize_task = Task(
    description="Summarize the transcribed text file into a concise summary.",
    expected_output="The summary should be saved as a text file.",
    agent=summarize_agent,
)

# Create and kickoff the crew
crew = Crew(
    agents=[download_agent, transcribe_agent, summarize_agent],
    tasks=[download_task, transcribe_task, summarize_task],
    process=Process.sequential,
)

result = crew.kickoff(inputs={"url": "https://www.youtube.com/watch?v=aywZrzNaKjs"})
print(result)
