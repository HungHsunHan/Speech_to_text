import os
import subprocess

import whisper
import yt_dlp


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


def transcribe_audio(audio_path):
    model = whisper.load_model(
        "base"
    )  # You can choose "base", "small", "medium", or "large"
    try:
        result = model.transcribe(audio_path, language="zh")
        return result["text"]
    except RuntimeError as e:
        print(f"Error transcribing audio: {str(e)}")
        raise


def main():
    # YouTube video URL
    video_url = input("Enter the YouTube video URL: ")
    video_name = input("Enter the Name of file: ")

    # Output paths
    base_path = os.path.join(os.getcwd(), "youtube_output")
    os.makedirs(base_path, exist_ok=True)

    video_path = os.path.join(base_path, video_name)

    try:
        print("Downloading video and extracting audio...")
        download_youtube_video(video_url, video_path)

        print("Transcribing audio...")
        transcription = transcribe_audio(video_path + ".wav")

        print("Transcription:")
        print(transcription)

        # Save the transcription to a file
        transcription_path = os.path.join(base_path, "transcription.txt")
        with open(transcription_path, "w", encoding="utf-8") as f:
            for i in range(0, len(transcription), 20):
                line = transcription[i : i + 20]
                f.write(line)

        print(f"Transcription saved to: {transcription_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
