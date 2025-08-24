# パート 3: 構造化出力、関数呼び出し、ネイティブツール

このセクションでは、Gemini API の 3 つの強力な機能について説明します。情報を定義されたスキーマに抽出するための構造化出力、外部ツールや API に接続するための関数呼び出し、そして Google 検索などのネイティブツールによる機能強化です。


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

## 1. 構造化出力

構造化出力を使用すると、Gemini が非構造化テキストの代わりに特定の形式の JSON で応答するように制約できます。これは、以下の目的で不可欠です。
- **データ抽出**: 非構造化テキストを構造化データに変換します
- **API 連携**: 後続処理のために一貫したフォーマットを取得します
- **データベースへの挿入**: データがスキーマ要件に一致することを確認します
- **品質管理**: 応答に必要なフィールドが含まれていることを検証します


```python
class Recipe(BaseModel):
    recipe_name: str
    ingredients: List[str]
    prep_time_minutes: int
    difficulty: str  # "easy", "medium", "hard"
    servings: int

class RecipeList(BaseModel):
    recipes: List[Recipe]

# 構造化出力に Pydantic モデルを使用
response = client.models.generate_content(
    model=MODEL_ID,
    contents="人気のクッキーのレシピを 2 つ、材料と準備の詳細を教えてください。",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=RecipeList,
    ),
)

# 構造化データを直接取得
recipes: List[Recipe] = response.parsed
for recipe in recipes.recipes:
    print(f"レシピ: {recipe.recipe_name}")
    print(f"材料: {recipe.ingredients}")
    print(f"準備時間: {recipe.prep_time_minutes} 分")
    print(f"難易度: {recipe.difficulty}")
    print(f"分量: {recipe.servings}")
    print("\n")
```

## !! 演習: PDF から構造化データへ !!

Files API と構造化出力を使用して、PDF の請求書またはドキュメントから構造化情報を抽出します。

タスク:
- Pydantic スキーマ (`InvoiceItem` と `InvoiceData`) とサンプル PDF ファイルのパス (`../assets/data/rewe_invoice.pdf`) が提供されています。これを使用するか、独自の PDF 請求書に置き換えることができます。
- `client.files.upload()` を使用して PDF ファイルをアップロードします。
- `client.models.generate_content()` を呼び出します。
- `response.parsed` から解析された構造化データにアクセスします。


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

# PDF ファイルをアップロード (PDF パスに置き換えてください)
pdf_file_path = "../assets/data/rewe_invoice.pdf"

# TODO:
```

## 2. 関数呼び出し

関数呼び出しを使用すると、Gemini は定義した特定の関数をいつ呼び出すかをインテリジェントに決定できます。これにより、次のことが可能になります。
- **外部 API 連携**: 天気、株価、データベースに接続します
- **動的な計算**: リアルタイムの計算を実行します
- **システムとの対話**: コマンドを実行したり、システム情報を取得したりします
- **複数ステップのワークフロー**: 複雑なタスクのために一連の関数呼び出しを連鎖させます


```python
def get_weather(location: str) -> dict:
    """指定された場所の現在の天気を取得します。
    
    Args:
        location: 都市名 (例: "San Francisco")
        
    Returns:
        天気情報の辞書
    """
    # モックの天気データ - 実際の使用では、天気 API を呼び出します
    weather_data = {
        "temperature": 22,
        "condition": "sunny", 
        "humidity": 60,
        "location": location,
        "feels_like": 24
    }
    print(f"🌤️ 関数が呼び出されました: get_weather(location='{location}')")
    return weather_data

# モデルの関数宣言を定義
weather_function = {
    "name": "get_weather",
    "description": "指定された場所の現在の天気を取得します",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "都市名"
            }
        },
        "required": ["location"]
    }
}
tools = [types.Tool(function_declarations=[weather_function])]

# ユーザープロンプトを定義
contents = [
    types.Content(
        role="user", parts=[types.Part(text="東京の天気は？")]
    )
]


# 関数宣言付きでリクエストを送信
response = client.models.generate_content(
    model=MODEL_ID,
    contents=contents,
    config=types.GenerateContentConfig(tools=tools)
)

# 関数呼び出しを確認
function_call = response.candidates[0].content.parts[0].function_call
print(f"モデルが呼び出したい関数: {function_call.name}")
print(f"引数: {dict(function_call.args)}")
```


```python
# 関数を実行
if function_call.name == "get_weather":
    result = get_weather(**function_call.args)
else:
    result = {"error": "不明な関数"}

print(f"関数の結果: {result}")

# 関数の結果をモデルに送り返す
function_response_part = types.Part.from_function_response(
    name=function_call.name,
    response={"result": result}
)
# contents に関数呼び出しと関数実行の結果を追加
contents.append(types.Content(role="model", parts=[types.Part(function_call=function_call)])) # モデルの関数呼び出しメッセージを追加
contents.append(types.Content(role="user", parts=[function_response_part])) # 関数の応答を追加

# 最終的な応答を取得
final_response = client.models.generate_content(
    model=MODEL_ID,
    contents=contents,
    config=types.GenerateContentConfig(tools=tools)
)

print(f"\n最終的な応答: {final_response.text}")
```

### 自動関数呼び出し (Python のみ)

Python SDK は、関数の実行を自動的に処理できます。


```python
def calculate_area(length: float, width: float) -> dict:
    """長方形の面積を計算します。
    
    Args:
        length: 長方形の長さ
        width: 長方形の幅

    Returns:
        価格計算
    """
    area = length * width
    print(f"計算: {length} × {width} = {area}")
    return {"operation": "area", "result": area}

# 自動関数呼び出しを使用 - はるかに簡単です！
config = types.GenerateContentConfig(
    tools=[get_weather, calculate_area]  # 関数を直接渡す
)

response = client.models.generate_content(
    model=MODEL_ID,
    contents="東京の天気と、5x3 メートルの部屋の面積は？",
    config=config
)

print(response.text)  # SDK が関数呼び出しを自動的に処理します
```

## !! 演習: 電卓エージェント !!

一連の電卓関数 (加算、減算、乗算、除算) を作成し、Gemini の関数呼び出し機能を使用して、自然言語のプロンプトに基づいて計算を実行します。

タスク:
- `add(a: float, b: float)`、`subtract(a: float, b: float)`、`multiply(a: float, b: float)`、`divide(a: float, b: float)` の Python 関数を定義します。
- これらの関数のリスト (`calculator_tools`) を作成します。
- `client.models.generate_content()` を使用して単一の操作をテストします。
- 複数ステップの計算のプロンプト (例: 「(25 + 15) * 3 - 10 を計算してください。これをステップバイステップで実行してください。」) を使用して複雑な式をテストします。


```python
# TODO:
```

## 3. ネイティブツール

Gemini は、Web の検索や URL コンテンツの分析など、機能を強化するためのネイティブツールを提供します。

### Google 検索連携

**ユースケース:**
- 最新の出来事とニュース
- リアルタイムのデータ検索
- 事実確認
- 調査支援


```python
# Google 検索ツールを定義
google_search_tool = types.Tool(google_search=types.GoogleSearch())

# 最新の出来事に関するクエリ
response = client.models.generate_content(
    model=MODEL_ID,
    contents="2025 年の再生可能エネルギー技術の最新動向は何ですか？",
    config=types.GenerateContentConfig(
        tools=[google_search_tool],
    )
)

print("🔍 最新の再生可能エネルギーニュース:")
print(response.text)
```

### URL コンテキストツール

**ユースケース:**
- Web サイトのコンテンツ分析
- ドキュメントの要約
- 競合調査
- コンテンツ抽出


```python
# 特定の Web ページを分析するための URL コンテキスト
url_context_tool = types.Tool(url_context=types.UrlContext())

response = client.models.generate_content(
    model=MODEL_ID,
    contents="https://www.python.org/about/ に記載されている主な機能と利点を 3 つの箇条書きで要約してください。",
    config=types.GenerateContentConfig(
        tools=[url_context_tool],
    )
)

print("🌐 Python.org の概要:")
print(response.text)
```

### コード実行ツール

Gemini は Python コードを実行して、計算、視覚化の作成、データの処理を行うことができます。


```python
# コード実行ツール
code_execution_tool = types.Tool(code_execution={})

response = client.models.generate_content(
    model=MODEL_ID,
    contents="世界の人口上位 5 都市の人口を示す棒グラフを作成してください。matplotlib を使用してください。",
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

## !! 演習: コード実行によるデータ分析 !!

Google 検索とコード実行ツールを組み合わせて、実世界のデータを見つけ、Gemini が生成・実行した Python コードを使用してそのデータを分析または視覚化します。

タスク:
- 情報の検索とその情報の処理/視覚化の両方を必要とするプロンプトを定義します。例: 「世界の人口上位 5 都市の人口を検索し、その人口の棒グラフを作成してください。」
- コード実行用の `types.Tool` を作成します: `code_execution_tool = types.Tool(code_execution={})`。
- Google 検索用の `types.Tool` を作成します: `google_search_tool = types.Tool(google_search=types.GoogleSearch())`。
- `client.models.generate_content()` を呼び出します。
- `response.candidates[0].content.parts` を反復処理し、各パートを表示します。


```python
# TODO:
```

## まとめと次のステップ

**学習したこと:**
- 信頼性の高いデータ抽出と検証のための Pydantic モデルを使用した構造化出力
- 外部 API、データベース、カスタムビジネスロジックを統合するための関数呼び出し
- Google 検索、URL コンテキスト分析、コード実行などのネイティブツール
- 包括的なワークフローと複雑な問題解決のための複数のツールの組み合わせ

**重要なポイント:**
- 構造化出力により、下流のアプリケーションで一貫したデータ形式が保証されます
- 関数呼び出しにより、外部システムやリアルタイムデータとのシームレスな統合が可能になります
- ネイティブツールは、追加のセットアップやインフラストラクチャなしで強力な機能を提供します
- ツールの組み合わせにより、高度なワークフローと複数ステップの問題解決が可能になります
- 信頼性の高いツール操作には、適切な検証とエラー処理が不可欠です

**次のステップ:** [パート 4: モデルコンテキストプロトコル (MCP)](https://github.com/philschmid/gemini-2.5-ai-engineering-workshop/blob/main/notebooks/04-model-context-protocol-mcp.ipynb) に進んでください [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philschmid/gemini-2.5-ai-engineering-workshop/blob/main/notebooks/04-model-context-protocol-mcp.ipynb)

**その他のリソース:**
- [構造化出力のドキュメント](https://ai.google.dev/gemini-api/docs/structured-output?lang=python)
- [関数呼び出しのドキュメント](https://ai.google.dev/gemini-api/docs/function-calling?lang=python)
- [Google 検索によるグラウンディング](https://ai.google.dev/gemini-api/docs/grounding)
- [URL コンテキストツール](https://ai.google.dev/gemini-api/docs/url-context)
- [コード実行のドキュメント](https://ai.google.dev/gemini-api/docs/code-execution)

```
