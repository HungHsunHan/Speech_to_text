import os

import whisper

from speech_to_text import convert_audio, download_youtube_video

video_url = "https://www.youtube.com/watch?v=RbRMuTk8A4A"

# Output paths
base_path = os.path.join(os.getcwd(), "youtube_output")
os.makedirs(base_path, exist_ok=True)
video_path = os.path.join(base_path, "video2")
download_youtube_video(video_url, video_path)


path = "/Users/hanhongxun/Desktop/git-repos/speech_to_text/youtube_output/video2.wav"
model = whisper.load_model("base")
result = model.transcribe(path)
print(result["text"])
