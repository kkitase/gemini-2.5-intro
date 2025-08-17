[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philschmid/gemini-2.5-ai-engineering-workshop/blob/main/notebooks/02-multimodal-capabilities.ipynb)

# Part 2 - Multimodal Capabilities (Images, Audio, Video, Documents)

Gemini models like `gemini-2.5-flash-preview-05-20` can process text, images, audio, video, and documents in a single prompt using `client.models.generate_content()`. This enables powerful multimodal AI applications that can understand and generate content across different media types.

**Key Capabilities:**
- **Visual Understanding**: Analyze images, extract text, identify objects
- **Audio Processing**: Transcribe speech, analyze music, understand audio content
- **Video Analysis**: Summarize videos, extract key frames, understand motion
- **Document Processing**: Extract information from PDFs, understand layouts
- **Multimodal Generation**: Create images and speech from text prompts


```python
%pip install pillow
```


```python
from google import genai
from google.genai import types
import os
import sys
import requests
from PIL import Image
from io import BytesIO

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import userdata
    GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')
else:
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY',None)

# Create client with api key
MODEL_ID = "gemini-2.5-flash-preview-05-20"
client = genai.Client(api_key=GEMINI_API_KEY)
```

## 1. Image Understanding: Single Image

Gemini can analyze images in multiple formats: PIL `Image` objects, raw bytes, or uploaded files via the File API.

**When to use each method:**
- **Raw bytes**: When working with image data from APIs or memory
- **File API**: Large images (>20MB), when you want to reuse images across multiple requests


```python
!curl -o image.jpg "https://storage.googleapis.com/generativeai-downloads/images/Cupcakes.jpg"
```


```python
with open('image.jpg', 'rb') as f:
    image_bytes = f.read()

prompt_specific = "Are there any fruits visible?"

response_specific = client.models.generate_content(
    model=MODEL_ID,
    contents=["What is this image?",
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")]
)
print(response_specific.text)
```

You can use the File API for large payloads (>20MB).

> The File API lets you store up to 20 GB of files per project, with a per-file maximum size of 2 GB. Files are stored for 48 hours. They can be accessed in that period with your API key, but cannot be downloaded from the API. It is available at no cost in all regions where the Gemini API is available.


```python
file_id = client.files.upload(file="../assets/data/Cupcakes.jpg")

response = client.models.generate_content(
    model=MODEL_ID,
    contents=["What is this image?", file_id]
)

print(response.text)
```

> The File API lets you store up to 20 GB of files per project, with a per-file maximum size of 2 GB. Files are stored for 48 hours. They can be accessed in that period with your API key, but cannot be downloaded from the API. It is available at no cost in all regions where the Gemini API is available.

## 2. Image Understanding: Multiple Images

Gemini can analyze and compare multiple images simultaneously, which is powerful for comparative analysis, visual storytelling, or understanding sequences.


```python
image_url_1 = "https://plus.unsplash.com/premium_photo-1694819488591-a43907d1c5cc?fm=jpg&q=60&w=3000&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8Y3V0ZSUyMGRvZ3xlbnwwfHwwfHx8MA%3D%3D" # Dog
image_url_2 = "https://images.pexels.com/photos/2071882/pexels-photo-2071882.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500" # Cat

image_response_req_1 = requests.get(image_url_1)
image_response_req_2 = requests.get(image_url_2)


# TODO: Call client.models.generate_content() to compare the two images.
# The contents should be a list containing:
# 1. A text part: "Compare these two images. What are the main subjects in each, and what are they doing?"
# 2. A text part: "Image 1:"
# 3. Image 1 bytes as a Part: types.Part.from_bytes(data=image_response_req_1.content, mime_type="image/jpeg")
# 4. A text part: "Image 2:"
# 5. Image 2 bytes as a Part: types.Part.from_bytes(data=image_response_req_2.content, mime_type="image/jpeg")
# response_multi = client.models.generate_content(
#     model=MODEL_ID,
#     contents=[
#         ...,
#         ..., ..., 
#         ..., ...
#     ]
# )
# print(response_multi.text)
```

## 3. !! Exercise: Product Description from Image !!

Use Gemini to analyze an image of a product and generate a detailed description, including features, use cases, and a marketing slogan.

Tasks:
- Find an image URL of a product (e.g., a backpack, a mug, a piece of electronics).
- Use the `requests` library to get the image content from the URL.
- Create a `types.Part` object from the image bytes.
- Create a text `types.Part` object containing a prompt that asks the model about the Product. 
- Call `client.models.generate_content()` with the `MODEL_ID` and a list containing your text prompt part and the image part.


```python
# TODO: 
```

## 4. Audio Understanding

Gemini can process audio files for transcription, content analysis, speaker identification, and audio summarization. This is particularly useful for podcasts, meetings, interviews, and voice memos.

**Supported audio formats**: MP3, WAV, FLAC, AAC, and other common formats


```python
file_path = "../assets/data/audio.mp3"

file_id = client.files.upload(file=file_path)

# Generate a structured response using the Gemini API
prompt = """Generate a transcript of the episode. Include timestamps and identify speakers.

Speakers:
- John

eg:
[00:00] Brady: Hello there.
[00:02] Tim: Hi Brady.

It is important to include the correct speaker names. Use the names you identified earlier. If you really don't know the speaker's name, identify them with a letter of the alphabet, eg there may be an unknown speaker 'A' and another unknown speaker 'B'.

If there is music or a short jingle playing, signify like so:
[01:02] [MUSIC] or [01:02] [JINGLE]

If you can identify the name of the music or jingle playing then use that instead, eg:
[01:02] [Firework by Katy Perry] or [01:02] [The Sofa Shop jingle]

If there is some other sound playing try to identify the sound, eg:
[01:02] [Bell ringing]

Each individual caption should be quite short, a few short sentences at most.

Signify the end of the episode with [END].

Don't use any markdown formatting, like bolding or italics.

Only use characters from the English alphabet, unless you genuinely believe foreign characters are correct.

It is important that you use the correct words and spell everything correctly. Use the context of the podcast to help.
If the hosts discuss something like a movie, book or celebrity, make sure the movie, book, or celebrity name is spelled correctly."""
audio_part = types.Part.from_uri(file_uri=file_id.uri, mime_type=file_id.mime_type)

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt, audio_part]
)
print(response.text)
```

## 5. Video Understanding

Gemini can process video files to understand their content, analyze scenes, identify objects and actions, and provide detailed summaries.

**Video capabilities:**
- Scene analysis and summarization
- Object and action recognition
- Temporal understanding (what happens when)
- Content extraction and key moments
- YouTube video analysis


```python
from time import sleep

video_path = "../assets/data/standup.mp4"

video_file_id = client.files.upload(file=video_path)
def wait_for_file_ready(file_id):
    while file_id.state == "PROCESSING":
        sleep(1)
        file_id = client.files.get(name=file_id.name)
        wait_for_file_ready(file_id)
    return file_id

video_file_id = wait_for_file_ready(video_file_id)


prompt = "Describe the main events in this video. What is the primary subject?"
video_part = types.Part.from_uri(file_uri=video_file_id.uri, mime_type=video_file_id.mime_type)



# TODO: Call client.models.generate_content() to analyze the video.
# The contents should be a list containing the prompt and video_part.
# response = client.models.generate_content(
#     model=MODEL_ID,
#     contents=[..., ...]
# )
# print(response.text)
```

### YouTube Video Analysis

The Gemini API supports direct YouTube URL analysis, which is very convenient for content analysis:


```python
# Analyze a YouTube video directly
youtube_url = "https://www.youtube.com/watch?v=dwgmfSOZNoQ"  # Google Cloud Next '25 Opening Keynote

youtube_part = genai.types.Part(
    file_data=genai.types.FileData(file_uri=youtube_url)
)
prompt = "What was the biggest Gemini announcement in this video?"

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt, youtube_part]
)

print(response.text)
```

## 6. !! Exercise: Summarize a YouTube Video !!

Analyze a YouTube video using its URL and generate a summary or transcript.

Tasks:
- Find a YouTube video URL (e.g., a tutorial, news segment, or educational video).
- Create a `genai.types.Part` object from the YouTube URL. You can use `genai.types.Part(file_data=genai.types.FileData(file_uri=youtube_url))` for this.
- Define a prompt asking the model to perform a task, such as summarizing the video.
- Call `client.models.generate_content()` with the `MODEL_ID`, your prompt, and the YouTube video part.


```python
# TODO:
```

**Try these variations:**
- Analyze a tutorial video and extract step-by-step instructions
- Summarize a news video and identify key facts vs. opinions
- Analyze a product review and extract pros/cons
- Process an educational video and create study notes

## 7. Working with PDF/Document Files

Gemini can extract information from PDFs and other document formats, making it excellent for document analysis, data extraction, and content summarization.

**Common use cases:**
- Invoice processing and data extraction
- Contract analysis and summarization
- Research paper analysis
- Form processing and validation
- Document classification and routing


```python
pdf_file_path = "../assets/data/rewe_invoice.pdf"

pdf_file_id = client.files.upload(file=pdf_file_path)

prompt = "What is the total amount due?"
pdf_part = types.Part.from_uri(file_uri=pdf_file_id.uri, mime_type=pdf_file_id.mime_type)

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt, pdf_part]
)
print(response.text)
```

## 7. Code 

Gemini is good at understanding and generating code. Let's use [gitingest](https://github.com/cyclotruc/gitingest) to chat with a GitHub repo:


```python
%pip install gitingest
```


```python
from gitingest import ingest_async

summary, tree, content = await ingest_async("https://github.com/philschmid/nextjs-gemini-2-0-pdf-structured-data")
```


```python
print(summary)
```


```python
print(tree)
```


```python
prompt = f"""Explain what repository is about:

Code:
{content}
"""

chat = client.chats.create(model=MODEL_ID)

response = chat.send_message(prompt)
print(response.text)
```


```python
response = chat.send_message("How are the schemas defined?")
print(response.text)
```


```python
response = chat.send_message("Update all schema route to use the new Gemini 2.5 models, `gemini-2.5-flash-preview-05-20`. Return only the updated file.")
print(response.text)
```

## 9. Image Generation

Generate high-quality images using Gemini's image generation capabilities. This feature is perfect for creating visual content, prototypes, marketing materials, and creative projects.

**Image Generation Features:**
- Text-to-image generation
- Style control through prompts
- High-resolution output
- SynthID watermarking for authenticity
- Multiple aspect ratios and sizes


```python
from PIL import Image
from io import BytesIO


prompt_text = "A photo of a cat"

response = client.models.generate_content(
    model="gemini-2.0-flash-preview-image-generation",
    contents=prompt_text,
    config=types.GenerateContentConfig(
      response_modalities=['TEXT', 'IMAGE']
    )
)

# Process the response
image_saved = False
for part in response.candidates[0].content.parts:
  if part.text is not None:
    print(f"Text response: {part.text}")
  elif part.inline_data is not None and part.inline_data.mime_type.startswith('image/'):
      image = Image.open(BytesIO(part.inline_data.data))
      image_filename = 'gemini_generated_image.png'
      image.save(image_filename)

image
```

**Image Generation Tips:**
- Be specific about style (photorealistic, illustration, cartoon, etc.)
- Include lighting and mood descriptors
- Specify composition details (close-up, wide shot, etc.)
- Mention art styles or references when relevant
- Consider aspect ratio and resolution needs

> **Note**: All generated images include a SynthID watermark for authenticity verification. More details in the [official documentation](https://ai.google.dev/gemini-api/docs/image-generation).

## 10. Text to Speech

Convert text into natural-sounding speech with controllable voice characteristics. This feature enables creating audio content, accessibility features, and interactive applications.

**TTS Capabilities:**
- Multiple voice options and styles
- Controllable pace, tone, and emotion
- Single-speaker and multi-speaker audio
- High-quality audio output
- Natural language voice direction

For this example, we'll use the `gemini-2.5-flash-preview-tts` model to generate single-speaker audio. You'll need to set the `response_modalities` to `["AUDIO"]` and provide a `SpeechConfig`.


```python
%pip install soundfile numpy
```


```python
import soundfile as sf
import numpy as np
from IPython.display import Audio, display

text_to_speak = "Say cheerfully: AI Eingeering Worlds Fair is the best conference in the world!"

response_tts = client.models.generate_content(
   model="gemini-2.5-flash-preview-tts", # Specific model for TTS
   contents=text_to_speak,
   config=types.GenerateContentConfig(
      response_modalities=["AUDIO"],
      speech_config=types.SpeechConfig(
         voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
               voice_name='Kore', # Choose from available prebuilt voices
            )
         )
      ),
   )
)

audio_array = np.frombuffer(response_tts.candidates[0].content.parts[0].inline_data.data, dtype=np.int16)
sf.write("generated_speech.wav", audio_array, 24000)
display(Audio("generated_speech.wav"))
```

## !! Exercise: Avatar Generation !!


Combine image generation and text-to-speech capabilities to create a visual avatar and an audio introduction for it.


1.  **Generate an Avatar Image:**
    - Write a descriptive prompt for an avatar image (e.g., "A friendly, futuristic robot assistant with a welcoming smile, digital art style, high resolution").
    - Use `client.models.generate_content()` with the model `gemini-2.0-flash-preview-image-generation`.
    - Set `response_modalities=['TEXT', 'IMAGE']` in `GenerateContentConfig`.
    - Process the response to extract the image data (from `part.inline_data.data` where `mime_type` starts with `image/`).
    - Save the image (e.g., as `generated_avatar.png`) using `PIL.Image` and `BytesIO`.
    - Display the generated image.
2.  **Create an Introduction Text:**
    - Write a short introductory sentence for your avatar (e.g., "Hello! I am Vision, your friendly AI assistant. I'm excited to help you generate amazing things!").
3.  **Generate Speech for the Introduction:**
    - Use `client.models.generate_content()` with the model `gemini-2.5-flash-preview-tts`.
    - For the `contents`, you can augment the introduction text with a description of the avatar to influence the voice (e.g., f"Say in a voice based on this image description {{your_image_prompt}}: {{your_introduction_text}}").
    - Configure `GenerateContentConfig` with `response_modalities=["AUDIO"]`.
    - Set up `speech_config` within the `GenerateContentConfig` to select a `prebuilt_voice_config` (e.g., `voice_name='Puck'`).
    - Process the response to get the audio data (from `part.inline_data.data`).
    - Convert the audio data to a NumPy array and save it as a WAV file (e.g., `avatar_introduction.wav`) using `soundfile`.
    - Provide a way to play the audio (e.g., `IPython.display.Audio`).


```python
# TODO:
```

## Recap & Next Steps

**What You've Learned:**
- Image understanding with single and multiple image analysis for various use cases
- Audio processing including speech transcription and audio content analysis
- Video analysis for scene understanding and YouTube content processing
- Document processing with PDF analysis and structured data extraction
- Code understanding for repository analysis and code review
- Creative generation with image creation and text-to-speech synthesis
- Multimodal integration combining different content types for rich applications
- File API usage for efficient handling of large files and reusable content

**Key Takeaways:**
- Use File API for large files (>20MB) and content you'll reuse multiple times
- Implement comprehensive error handling for network and API operations
- Structure prompts clearly and specifically for consistent, high-quality outputs
- Monitor token usage across different modalities for effective cost control
- Consider user experience and processing time for multimedia operations

**Next Steps:** Continue with [Part 3: Structured Outputs, Function Calling & Tools](https://github.com/philschmid/gemini-2.5-ai-engineering-workshop/blob/main/notebooks/03-structured-outputs-function-calling-tools.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philschmid/gemini-2.5-ai-engineering-workshop/blob/main/notebooks/03-structured-outputs-function-calling-tools.ipynb)

**More Resources:**
- [Vision Understanding Documentation](https://ai.google.dev/gemini-api/docs/vision?lang=python)
- [Audio Understanding Documentation](https://ai.google.dev/gemini-api/docs/audio?lang=python)
- [Image Generation Guide](https://ai.google.dev/gemini-api/docs/image-generation)
- [Text-to-Speech Documentation](https://ai.google.dev/gemini-api/docs/speech-generation)
