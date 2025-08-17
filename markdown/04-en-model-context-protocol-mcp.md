[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philschmid/gemini-2.5-ai-engineering-workshop/blob/main/notebooks/04-model-context-protocol-mcp.ipynb)

# Part 4: Model Context Protocol (MCP)

The Model Context Protocol (MCP) is an open standard for connecting AI assistants to external data sources and tools. It enables seamless integration between LLMs and various services, databases, and APIs through a standardized protocol.


```python
%pip install mcp
```


```python
from google import genai
from google.genai import types
import sys
import os
import asyncio
from datetime import datetime
from mcp import ClientSession, StdioServerParameters
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.stdio import stdio_client

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

## What is MCP?

Model Context Protocol (MCP) is a revolutionary approach to extending AI capabilities. Unlike traditional function calling where you define functions locally in your code, MCP allows AI models to connect to remote servers that provide tools and resources.


- **🔌 Plug-and-Play Integration**: Connect to any MCP-compatible service instantly
- **🌐 Remote Capabilities**: Access tools and data from anywhere on the internet
- **🔄 Standardized Protocol**: One protocol works with all MCP servers
- **🔒 Centralized Security**: Control access and permissions at the server level
- **📈 Scalability**: Share resources across multiple AI applications
- **🛠️ Rich Ecosystem**: Growing library of MCP servers for various use case

## 1. Working with Stdio MCP Servers

Stdio (Standard Input/Output) servers run as local processes and communicate through pipes. This is perfect for:
- Development and testing
- Local tools and utilities
- Lightweight integrations


## 1. Working with MCP Servers

Let's use the DeepWiki MCP server, which provides access to Wikipedia data and search capabilities:


```python
# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="npx",  # Executable
    args=["-y", "@philschmid/weather-mcp"],  # MCP Server
    env=None,  # Optional environment variables
)

async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Prompt to get the weather for the current day in London.
            prompt = f"What is the weather in London in {datetime.now().strftime('%Y-%m-%d')}?"
            # Initialize the connection between client and server
            await session.initialize()
            # Send request to the model with MCP function declarations
            response = await client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0,
                    tools=[session],  # uses the session, will automatically call the tool
                    # Uncomment if you **don't** want the sdk to automatically call the tool
                    # automatic_function_calling=genai.types.AutomaticFunctionCallingConfig(
                    #     disable=True
                    # ),
                ),
            )
            print(response.text)

await run()
```

## !! Exercise: Build Your Own MCP CLI Agent !!

Create an interactive command-line interface (CLI) chat agent that connects to the DeepWiki MCP server (a remote server providing access to Wikipedia-like data). The agent should allow users to ask questions about GitHub repositories, and it will use the DeepWiki server to find answers.

Task:
- Use `mcp.client.streamable_http.streamablehttp_client` to establish a connection to the remote URL (https://mcp.deepwiki.com/mcp). 
- Inside the `async with streamablehttp_client(...)` block, create an `mcp.ClientSession`.
- Initialize the session using `await session.initialize()`.
- Create a `genai.types.GenerateContentConfig` with `temperature=0` and pass the `session` object in the `tools` list. This configures the chat to use the MCP server.
- Create an asynchronous chat session using `client.aio.chats.create()`, passing the `MODEL_ID` (e.g., "gemini-2.5-flash-preview-05-20") and the `config` you created.
- Implement an interactive loop to chat with the model using `input()` to get the user's input.


```python
# TODO: 
```

## Recap & Next Steps

**What You've Learned:**
- Understanding the Model Context Protocol (MCP) and its advantages over traditional function calling
- Connecting to remote MCP servers using both stdio and HTTP protocols
- Building interactive chat agents that leverage MCP capabilities

**Key Takeaways:**
- MCP enables plug-and-play integration with external services and data sources
- Remote capabilities provide access to tools and data from anywhere on the internet
- Standardized protocols ensure compatibility across different AI applications
- Centralized security and permissions improve enterprise deployment scenarios
- The MCP ecosystem is rapidly growing with servers for various use cases

🎉 **Congratulations!** You've completed the Gemini 2.5 AI Engineering Workshop

**More Resources:**
- [MCP with Gemini Documentation](https://ai.google.dev/gemini-api/docs/function-calling?example=weather#model_context_protocol_mcp)
- [Function Calling Documentation](https://ai.google.dev/gemini-api/docs/function-calling?lang=python)
- [MCP Official Specification](https://spec.modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Server Directory](https://github.com/modelcontextprotocol/servers)
