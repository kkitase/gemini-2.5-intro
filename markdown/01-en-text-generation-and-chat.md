
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philschmid/gemini-2.5-ai-engineering-workshop/blob/main/notebooks/01-text-generation-and-chat.ipynb)

# Part 1 - Text Generation and Chat

This part focuses on text generation with the Gemini API using the `google-genai` SDK, including basic prompts, chat interactions, streaming, and configuration.

Make sure you have completed the [setup and authentication](solution_00_setup_and_authentication.md) section.

```python
from google import genai
from google.genai import types
import os
import sys
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

## 1. Send Your First Prompt

```python
prompt = "Create 3 names for a new coffee shop that emphasizes sustainability."

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt
)

print("Response from Gemini:")
print(response.text)
```

#### !! Exercise: Sending Various Prompts !!

Practice sending different types of prompts to the Gemini model and observe its responses. You can also experiment with different model versions if they are available to you.

Tasks:
- Write a prompt to ask Gemini to generate a short poem about a robot.
- Write a prompt to ask Gemini to explain "machine learning" in simple terms.
- Try other models (e.g., `gemini-2.0-flash`) and send your prompts to them and compare the results.

```python
# TODO:
```

## 2. Understanding and Counting Tokens

Tokens are the basic units that Gemini models use to process text. Understanding token usage is crucial for:
- **Cost management**: Billing is based on token consumption
- **Context limits**: Models have maximum token limits (e.g., 1M tokens for Gemini 2.5 Pro)
- **Performance optimization**: Smaller inputs generally process faster

For Gemini models, a token is equivalent to about 4 characters, and 100 tokens equals about 60-80 English words.

### Count tokens before generation

You can count tokens in your input before sending it to the model to estimate costs and ensure you stay within limits:

```python
prompt = "The quick brown fox jumps over the lazy dog."

# Count tokens in the input
# TODO: Call the client.models.count_tokens() method.
# Make sure to pass the MODEL_ID and the prompt.
# token_count = client.models.count_tokens(
#     model=...,
#     contents=...
# )
print(f"Input tokens: {token_count.total_tokens}")

# Estimate cost (example pricing - check current rates)
estimated_cost = token_count.total_tokens * 0.15 / 1_000_000
print(f"Estimated input cost: ${estimated_cost:.6f}")
```

### Count tokens after generation

After generating content, you can access detailed token usage information:

```python
prompt = "Write a haiku about artificial intelligence."

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt
)

print(f"Generated haiku:\n{response.text}\n")

# Access token usage metadata
usage = response.usage_metadata
print(f"Input tokens: {usage.prompt_token_count}")
print(f"Thought tokens: {usage.thoughts_token_count}")
print(f"Output tokens: {usage.candidates_token_count}")

# Calculate total estimated cost
total_cost = (usage.prompt_token_count * 0.15 + (usage.candidates_token_count + usage.thoughts_token_count) * 3.5) / 1_000_000
print(f"Total estimated cost: ${total_cost:.6f}")
```

## 3. Text Understanding with `contents`

The simplest way to generate text is to provide the model with a text-only prompt. `contents` can be a single prompt, a list of prompts, or a combination of multimodal inputs.

```python
response_capital = client.models.generate_content(
    model=MODEL_ID,
    contents="What is the capital of France?"
)
print(f"Q: What is the capital of France?\nA: {response_capital.text}")
```

```python
# TODO: Call the client.models.generate_content() method.
# For the contents, provide a list of strings:
# 1. "Create 3 names for a vegan restaurant"
# 2. "city: Berlin"
# response_restaurant_berlin = client.models.generate_content(
#     model=MODEL_ID,
#     contents=[...]
# )
print(f"\nVegan restaurant names in Berlin:\n{response_restaurant_berlin.text}")
```

## 4. Streaming Responses

Streaming allows you to receive responses incrementally as they're generated, providing a better user experience for long responses or real-time applications like chatbots.

**When to use streaming:**
- Interactive applications (chatbots, assistants)
- Long content generation
- Real-time user feedback
- Improved perceived performance

```python
prompt_long_story = "Write a short story about a brave knight and a friendly dragon."

print("Streaming response:")
for chunk in client.models.generate_content_stream(
    model=MODEL_ID,
    contents=prompt_long_story
):
    if chunk.text:  # Check if chunk has text content
        print(chunk.text, end="", flush=True)
print("\n")  # Add newline at the end
```

## 5. Chat (Multi-turn Conversations)

The SDK chat class provides an interface to keep track of conversation history. Behind the scenes it uses the same `generate_content` method.

```python
chat_session = client.chats.create(model=MODEL_ID)

user_message1 = "I'm planning a weekend trip. Any suggestions for a city break in Europe?"
print(f"User: {user_message1}")
response1 = chat_session.send_message(message=user_message1)
print(f"Model: {response1.text}\n")
```

```python
user_message2 = "I like history and good food. Not too expensive."
print(f"User: {user_message2}")
# TODO: Call the chat_session.send_message() method with user_message2.
# response2 = chat_session.send_message(message=...)
```

```python
# View conversation history
history = chat_session.get_history()
print(f"Total messages in conversation: {len(history)}")
```

## 6. System Instructions

System instructions let you define the model's behavior and personality. They're applied consistently throughout the conversation.

**Best practices for system instructions:**
- Be specific and clear
- Define the role and tone
- Include formatting preferences
- Set behavioral guidelines

```python
system_instruction_poet = "You are a renowned poet from the 17th century, specializing in sonnets. Respond in iambic pentameter and use eloquent, period-appropriate language."

response_poet = client.models.generate_content(
    model=MODEL_ID,
    contents="What are your thoughts on modern technology?",
    config=types.GenerateContentConfig(
        system_instruction=system_instruction_poet
    )
)
print(f"\nPoet model on modern tech:\n{response_poet.text}")
```

## 7. Generation Configuration

Customize the generation behavior using configuration parameters. Understanding these helps you fine-tune responses for your specific use case.

```python
# Configuration using dictionary
generation_config_dict = {
    "temperature": 0.2,      # Lower = more deterministic, higher = more creative
    "max_output_tokens": 2000, # Limit response length
    "top_p": 0.8,            # Nucleus sampling - diversity of token selection
    "top_k": 30,             # Consider top 30 most likely tokens

}

# TODO: Call client.models.generate_content()
# Pass the MODEL_ID, a prompt to "Write a very short tagline for a new brand of eco-friendly sneakers.",
# and the generation_config_dict.
# response_config = client.models.generate_content(
#     model=...,
#     contents=...,
#     config=...
# )
```

**Parameter Guide:**
- **Temperature (0.0-2.0)**: Controls randomness. Use 0.2-0.4 for factual content, 0.7-1.0 for creative content
- **Top-p (0.0-1.0)**: Controls diversity. Lower values = more focused, higher = more diverse
- **Top-k**: Limits token choices. Lower = more focused, higher = more diverse
- **Max output tokens**: Prevents overly long responses and controls costs

## 8. Long Context and File Uploads

Gemini 2.5 Pro has a 1M token context window. In practice, 1 million tokens could look like:

- 50,000 lines of code (with the standard 80 characters per line)
- All the text messages you have sent in the last 5 years
- 8 average length English novels
- 1 hour of video data

The File API allows you to upload files to the Gemini API and use them as context for your requests.

```python
# Example with a text file (more reliable than the audio example)
import requests

# Download a sample text file
sample_text_url = "https://www.gutenberg.org/files/74/74-0.txt"  # Adventures of Tom Sawyer
response_req = requests.get(sample_text_url)

# Save to local file
with open("sample_book.txt", "w", encoding="utf-8") as f:
    f.write(response_req.text)

# Upload the file to the Gemini API
try:
    myfile = client.files.upload(file="sample_book.txt")
    print(f"File uploaded successfully: {myfile.name}")
    
    # Generate content using the uploaded file as context
    response = client.models.generate_content(
        model=MODEL_ID, 
        contents=[myfile, "Summarize this book in 3 key points"])
    
    print("Summary:")
    print(response.text)
    
    # Check token usage for the large context
    print(f"\nToken usage: {response.usage_metadata.total_token_count}")
    
except Exception as e:
    print(f"Error uploading file: {e}")
    print("Make sure the file exists and is accessible")
```

## 9. !! Exercise: Chat with a "Book" !!

Create an interactive chat session where you can "talk" to the book "Alice in Wonderland". You'll set up the chat with a specific persona for the AI and use the book's text as context for the conversation.

Task: 
- Download the text of "Alice in Wonderland" (a helper code block is provided).
- Upload the book's text file (`alice_in_wonderland.txt`) to the Gemini API using `client.files.upload()`.
- Create a chat session using `client.chats.create()`:
- Send an initial message to the chat session using `chat.send_message()`:
- Send at least one follow-up question to the chat session (e.g., "Explain the various methods of speech delivery in more detail") and print its response.

```python
import requests

# Download Alice in Wonderland
book_text_url = "https://www.gutenberg.org/files/11/11-0.txt"
try:
    response_book_req = requests.get(book_text_url)
    response_book_req.raise_for_status()  # Raise an exception for bad status codes
    
    with open("alice_in_wonderland.txt", "w", encoding="utf-8") as f:
        f.write(response_book_req.text)
    print("Book downloaded successfully!")
    
except requests.RequestException as e:
    print(f"Error downloading book: {e}")
```

```python
# TODO:
```

```python
# TODO:
```

## Recap & Next Steps

**What You've Learned:**
- Basic text generation with `client.models.generate_content()` for single prompts
- Token counting and cost estimation for better resource management
- Streaming responses with `generate_content_stream()` for improved user experience
- Multi-turn conversations using `client.chats.create()` and chat sessions
- System instructions for consistent model behavior and personality
- Generation configuration parameters for fine-tuning responses
- Long context handling and file uploads with the File API
- Error handling and best practices for production applications

**Key Takeaways:**
- Monitor token usage to control costs and stay within limits
- Use streaming for interactive applications and long responses
- Configure parameters based on your use case (factual vs creative content)
- Implement proper error handling for robust applications
- System instructions are powerful for setting behavior and tone

**Next Steps:** Continue with [Part 2: Multimodal Capabilities](https://github.com/philschmid/gemini-2.5-ai-engineering-workshop/blob/main/notebooks/02-multimodal-capabilities.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philschmid/gemini-2.5-ai-engineering-workshop/blob/main/notebooks/02-multimodal-capabilities.ipynb)

**More Resources:**
- [Text Generation Guide](https://ai.google.dev/gemini-api/docs/text-generation)
- [Token Counting Guide](https://ai.google.dev/gemini-api/docs/tokens)
- [Long Context Documentation](https://ai.google.dev/gemini-api/docs/long-context)
- [File API Documentation](https://ai.google.dev/gemini-api/docs/files)
