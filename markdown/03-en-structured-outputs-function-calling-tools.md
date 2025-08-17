[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philschmid/gemini-2.5-ai-engineering-workshop/blob/main/notebooks/03-structured-outputs-function-calling-tools.ipynb)

# Part 3: Structured Outputs, Function Calling & Native Tools

This section covers three powerful capabilities of the Gemini API: structured outputs for extracting information into defined schemas, function calling for connecting to external tools and APIs, and native tools like Google Search for enhanced capabilities.


```python
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List, Optional
import sys
import os
from IPython.display import Image, Markdown

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

## 1. Structured Outputs

Structured outputs allow you to constrain Gemini to respond with JSON in a specific format instead of unstructured text. This is essential for:
- **Data extraction**: Converting unstructured text into structured data
- **API integration**: Getting consistent formats for downstream processing  
- **Database insertion**: Ensuring data matches your schema requirements
- **Quality control**: Validating that responses contain required fields


```python
class Recipe(BaseModel):
    recipe_name: str
    ingredients: List[str]
    prep_time_minutes: int
    difficulty: str  # "easy", "medium", "hard"
    servings: int

class RecipeList(BaseModel):
    recipes: List[Recipe]

# Using Pydantic models for structured output
response = client.models.generate_content(
    model=MODEL_ID,
    contents="Give me 2 popular cookie recipes with ingredients and prep details.",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=RecipeList,
    ),
)

# Get structured data directly
recipes: List[Recipe] = response.parsed
for recipe in recipes.recipes:
    print(f"Recipe: {recipe.recipe_name}")
    print(f"Ingredients: {recipe.ingredients}")
    print(f"Prep Time: {recipe.prep_time_minutes} minutes")
    print(f"Difficulty: {recipe.difficulty}")
    print(f"Servings: {recipe.servings}")
    print("\n")
```

## !! Exercise: PDF to Structured Data !!

Extract structured information from a PDF invoice or document using the Files API and structured outputs.

Tasks:
- A Pydantic schema (`InvoiceItem` and `InvoiceData`) and an example PDF file path (`../assets/data/rewe_invoice.pdf`) are provided. You can use this or replace it with your own PDF invoice.
- Upload the PDF file using `client.files.upload()`.
- Call `client.models.generate_content()`:
- Access the parsed structured data from `response.parsed`.


```python
class InvoiceItem(BaseModel):
    description: str
    quantity: int
    unit_price: float
    total: float

class InvoiceData(BaseModel):
    invoice_number: str
    date: str
    vendor_name: str
    vendor_address: str
    total_amount: float
    items: List[InvoiceItem]

# Upload a PDF file (replace with your PDF path)
pdf_file_path = "../assets/data/rewe_invoice.pdf"

# TODO:
```

## 2. Function Calling

Function calling allows Gemini to intelligently decide when to call specific functions you define. This enables:
- **External API integration**: Connect to weather, stocks, databases
- **Dynamic calculations**: Perform real-time computations
- **System interaction**: Execute commands or retrieve system information
- **Multi-step workflows**: Chain function calls for complex tasks


```python
def get_weather(location: str) -> dict:
    """Gets current weather for a location.
    
    Args:
        location: The city name, e.g. "San Francisco"
        
    Returns:
        Weather information dictionary
    """
    # Mock weather data - in real use, you'd call a weather API
    weather_data = {
        "temperature": 22,
        "condition": "sunny", 
        "humidity": 60,
        "location": location,
        "feels_like": 24
    }
    print(f"🌤️ FUNCTION CALLED: get_weather(location='{location}')")
    return weather_data

# Define function declarations for the model
weather_function = {
    "name": "get_weather",
    "description": "Gets current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city name"
            }
        },
        "required": ["location"]
    }
}
tools = [types.Tool(function_declarations=[weather_function])]

# Define user prompt
contents = [
    types.Content(
        role="user", parts=[types.Part(text="Whats the weather in Tokyo?")]
    )
]


# Send request with function declarations
response = client.models.generate_content(
    model=MODEL_ID,
    contents=contents,
    config=types.GenerateContentConfig(tools=tools)
)

# Check for function calls
function_call = response.candidates[0].content.parts[0].function_call
print(f"Model wants to call: {function_call.name}")
print(f"With arguments: {dict(function_call.args)}")
```


```python
# Execute the function
if function_call.name == "get_weather":
    result = get_weather(**function_call.args)
else:
    result = {"error": "Unknown function"}

print(f"Function result: {result}")

# Send function result back to model
function_response_part = types.Part.from_function_response(
    name=function_call.name,
    response={"result": result}
)
# Append function call and result of the function execution to contents
contents.append(types.Content(role="model", parts=[types.Part(function_call=function_call)])) # Append the model's function call message
contents.append(types.Content(role="user", parts=[function_response_part])) # Append the function response

# Get final response
final_response = client.models.generate_content(
    model=MODEL_ID,
    contents=contents,
    config=types.GenerateContentConfig(tools=tools)
)

print(f"\nFinal response: {final_response.text}")
```

### Automatic Function Calling (Python Only)

The Python SDK can automatically handle function execution for you:


```python
def calculate_area(length: float, width: float) -> dict:
    """Calculate the area of a rectangle.
    
    Args:
        length: Length of the rectangle
        width: Width of the rectangle

    Returns:
        Price calculations
    """
    area = length * width
    print(f"CALC: {length} × {width} = {area}")
    return {"operation": "area", "result": area}

# Using automatic function calling - much simpler!
config = types.GenerateContentConfig(
    tools=[get_weather, calculate_area]  # Pass functions directly
)

response = client.models.generate_content(
    model=MODEL_ID,
    contents="What's the weather in Tokyo and what's the area of a 5x3 meter room?",
    config=config
)

print(response.text)  # SDK handles function calls automatically
```

## !! Exercise: Calculator Agent !!

Create a set of calculator functions (add, subtract, multiply, divide) and use Gemini's function calling capability to perform calculations based on natural language prompts.

Tasks:
- Define Python functions for `add(a: float, b: float)`, `subtract(a: float, b: float)`, `multiply(a: float, b: float)`, and `divide(a: float, b: float)`.
- Create a list of these functions (`calculator_tools`).
- Test single operations with `client.models.generate_content()`:
- Test complex expressions with a prompt for a multi-step calculation (e.g., "Calculate (25 + 15) * 3 - 10. Do this step by step.").


```python
# TODO:
```

## 3. Native Tools

Gemini provides native tools for enhanced capabilities like searching the web and analyzing URL content.

### Google Search Integration

**Use cases:**
- Current events and news
- Real-time data lookup
- Fact verification
- Research assistance


```python
# Define Google Search tool
google_search_tool = types.Tool(google_search=types.GoogleSearch())

# Current events query
response = client.models.generate_content(
    model=MODEL_ID,
    contents="What are the latest developments in renewable energy technology in 2025?",
    config=types.GenerateContentConfig(
        tools=[google_search_tool],
    )
)

print("🔍 Current Renewable Energy News:")
print(response.text)
```

### URL Context Tool

**Use cases:**
- Website content analysis
- Documentation summarization
- Competitive research
- Content extraction


```python
# URL context for analyzing specific web pages
url_context_tool = types.Tool(url_context=types.UrlContext())

response = client.models.generate_content(
    model=MODEL_ID,
    contents="Summarize the key features and benefits mentioned on https://www.python.org/about/ in 3 bullet points.",
    config=types.GenerateContentConfig(
        tools=[url_context_tool],
    )
)

print("🌐 Python.org Summary:")
print(response.text)
```

### Code Execution Tool

Gemini can execute Python code to perform calculations, create visualizations, and process data.


```python
# Code execution tool
code_execution_tool = types.Tool(code_execution={})

response = client.models.generate_content(
    model=MODEL_ID,
    contents="Create a bar chart showing the population of the 5 largest cities in the world. Use matplotlib.",
    config=types.GenerateContentConfig(
        tools=[code_execution_tool],
    )
)


for p in response.candidates[0].content.parts:
    if p.text:
        display(Markdown(p.text))
    elif p.executable_code:
        display(Markdown(f"```python\n{p.executable_code.code}\n```"))
    elif p.inline_data:
        display(Image(data=p.inline_data.data, width=800, format="png"))

```

## !! Exercise: Data Analysis with Code Execution !!

Combine Google Search and Code Execution tools to find real-world data and then analyze or visualize it using Python code generated and executed by Gemini.

Tasks:
- Define a prompt that requires both searching for information and then processing/visualizing that information. For example: "Search for the population of the 5 largest cities in the world and create a bar chart of their populations."
- Create a `types.Tool` for code execution: `code_execution_tool = types.Tool(code_execution={})`.
- Create a `types.Tool` for Google Search: `google_search_tool = types.Tool(google_search=types.GoogleSearch())`.
- Call `client.models.generate_content()`:
- Iterate through the `response.candidates[0].content.parts` and display the parts.


```python
# TODO:
```

## Recap & Next Steps

**What You've Learned:**
- Structured outputs using Pydantic models for reliable data extraction and validation
- Function calling to integrate external APIs, databases, and custom business logic
- Native tools including Google Search, URL context analysis, and code execution
- Combining multiple tools for comprehensive workflows and complex problem-solving

**Key Takeaways:**
- Structured outputs ensure consistent data formats for downstream applications
- Function calling enables seamless integration with external systems and real-time data
- Native tools provide powerful capabilities without additional setup or infrastructure
- Tool combinations unlock sophisticated workflows and multi-step problem solving
- Proper validation and error handling are crucial for reliable tool interactions

**Next Steps:** Continue with [Part 4: Model Context Protocol (MCP)](https://github.com/philschmid/gemini-2.5-ai-engineering-workshop/blob/main/notebooks/04-model-context-protocol-mcp.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philschmid/gemini-2.5-ai-engineering-workshop/blob/main/notebooks/04-model-context-protocol-mcp.ipynb)

**More Resources:**
- [Structured Output Documentation](https://ai.google.dev/gemini-api/docs/structured-output?lang=python)
- [Function Calling Documentation](https://ai.google.dev/gemini-api/docs/function-calling?lang=python)
- [Grounding with Google Search](https://ai.google.dev/gemini-api/docs/grounding)
- [URL Context Tool](https://ai.google.dev/gemini-api/docs/url-context)
- [Code Execution Documentation](https://ai.google.dev/gemini-api/docs/code-execution)
