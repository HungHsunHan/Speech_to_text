import os
import subprocess

import whisper


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
        result = model.transcribe(audio_path)
        return result["text"]
    except RuntimeError as e:
        print(f"Error transcribing audio: {str(e)}")
        raise


def main():
    # YouTube video URL

    base_path = os.path.join(os.getcwd(), "yi")
    video_path = "/Users/hanhongxun/Desktop/20240714_113341.mp4"
    audio_path = os.path.join(os.getcwd(), "yi", "audio.wav")
    # import subprocess

    # command = f"ffmpeg -i {video_path} -ab 160k -ac 2 -ar 44100 -vn {os.path.join(base_path,'audio.wav')}"

    # subprocess.call(command, shell=True)
    try:
        # print("Downloading video and extracting audio...")
        # download_youtube_video(video_url, video_path)

        print("Transcribing audio...")
        transcription = transcribe_audio(audio_path)

        print("Transcription:")
        print(transcription)

        # Save the transcription to a file
        transcription_path = os.path.join(base_path, "transcription2.txt")
        with open(transcription_path, "w", encoding="utf-8") as f:
            for i in range(0, len(transcription), 20):
                line = transcription[i : i + 20]
                f.write(line)

        print(f"Transcription saved to: {transcription_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
