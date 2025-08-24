[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philschmid/gemini-2.5-ai-engineering-workshop/blob/main/notebooks/04-model-context-protocol-mcp.ipynb)

# パート 4: Model Context Protocol (MCP)

Model Context Protocol (MCP) は、AI アシスタントを外部のデータソースやツールに接続するためのオープンスタンダードです。標準化されたプロトコルを通じて、LLM とさまざまなサービス、データベース、API とのシームレスな統合を可能にします。


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

# API キーを使用してクライアントを作成
MODEL_ID = "gemini-2.5-flash-preview-05-20"
client = genai.Client(api_key=GEMINI_API_KEY)
```

## MCP とは？

Model Context Protocol (MCP) は、AI の能力を拡張するための画期的なアプローチです。コード内でローカルに機能を定義する従来の Function Calling とは異なり、MCP を使用すると、AI モデルはツールやリソースを提供するリモートサーバーに接続できます。


- **🔌 プラグアンドプレイ統合**: MCP 互換のあらゆるサービスに即座に接続
- **🌐 リモート機能**: インターネット上のどこからでもツールやデータにアクセス
- **🔄 標準化されたプロトコル**: 1 つのプロトコルがすべての MCP サーバーで動作
- **🔒 一元化されたセキュリティ**: サーバーレベルでアクセスと権限を制御
- **📈 スケーラビリティ**: 複数の AI アプリケーション間でリソースを共有
- **🛠️ 豊富なエコシステム**: さまざまなユースケースに対応する MCP サーバーのライブラリが拡大中

## 1. Stdio MCP サーバーの利用

Stdio (標準入出力) サーバーはローカルプロセスとして実行され、パイプを介して通信します。これは、次のような場合に最適です。
- 開発とテスト
- ローカルツールとユーティリティ
- 軽量な統合


## 1. MCP サーバーの利用

Wikipedia のデータと検索機能へのアクセスを提供する DeepWiki MCP サーバーを使ってみましょう。


```python
# stdio 接続用のサーバーパラメータを作成
server_params = StdioServerParameters(
    command="npx",  # 実行ファイル
    args=["-y", "@philschmid/weather-mcp"],  # MCP サーバー
    env=None,  # オプションの環境変数
)

async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # ロンドンの今日の天気を尋ねるプロンプト
            prompt = f"What is the weather in London in {datetime.now().strftime('%Y-%m-%d')}?"
            # クライアントとサーバー間の接続を初期化
            await session.initialize()
            # MCP 関数の宣言とともにモデルにリクエストを送信
            response = await client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0,
                    tools=[session],  # session を使用し、ツールを自動的に呼び出します
                    # SDK にツールを自動的に呼び出してほしくない場合は、コメントを解除してください
                    # automatic_function_calling=genai.types.AutomaticFunctionCallingConfig(
                    #     disable=True
                    # ),
                ),
            )
            print(response.text)

await run()
```

## !! 演習: 独自の MCP CLI エージェントを構築する !!

DeepWiki MCP サーバー (Wikipedia のようなデータへのアクセスを提供するリモートサーバー) に接続する、対話型のコマンドラインインターフェース (CLI) チャットエージェントを作成します。このエージェントを使用すると、ユーザーは GitHub リポジトリに関する質問をすることができ、DeepWiki サーバーを使用して回答を見つけます。

タスク:
- `mcp.client.streamable_http.streamablehttp_client` を使用して、リモート URL (https://mcp.deepwiki.com/mcp) への接続を確立します。
- `async with streamablehttp_client(...)` ブロック内で、`mcp.ClientSession` を作成します。
- `await session.initialize()` を使用してセッションを初期化します。
- `temperature=0` で `genai.types.GenerateContentConfig` を作成し、`tools` リストに `session` オブジェクトを渡します。これにより、チャットで MCP サーバーを使用するように構成されます。
- `client.aio.chats.create()` を使用して非同期チャットセッションを作成し、`MODEL_ID` (例: "gemini-2.5-flash-preview-05-20") と作成した `config` を渡します。
- `input()` を使用してユーザーの入力を取得し、モデルと対話するための対話ループを実装します。


```python
# TODO: 
```

## まとめと次のステップ

**学習したこと:**
- Model Context Protocol (MCP) と従来の Function Calling に対するその利点の理解
- stdio と HTTP プロトコルの両方を使用したリモート MCP サーバーへの接続
- MCP 機能を活用した対話型チャットエージェントの構築

**重要なポイント:**
- MCP は、外部サービスやデータソースとのプラグアンドプレイ統合を可能にします
- リモート機能により、インターネット上のどこからでもツールやデータにアクセスできます
- 標準化されたプロトコルにより、さまざまな AI アプリケーション間での互換性が保証されます
- 一元化されたセキュリティと権限により、エンタープライズでの導入シナリオが改善されます
- MCP エコシステムは、さまざまなユースケースに対応するサーバーで急速に成長しています

🎉 **おめでとうございます！** これで Gemini 2.5 AI エンジニアリングワークショップは完了です。

**その他のリソース:**
- [MCP と Gemini のドキュメント](https://ai.google.dev/gemini-api/docs/function-calling?example=weather#model_context_protocol_mcp)
- [Function Calling のドキュメント](https://ai.google.dev/gemini-api/docs/function-calling?lang=python)
- [MCP 公式仕様](https://spec.modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP サーバーディレクトリ](https://github.com/modelcontextprotocol/servers)
