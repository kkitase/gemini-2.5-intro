##### Copyright 2025 Google LLC.


# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Lyria RealTime を使った音楽生成入門

<a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Get_started_LyriaRealTime.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=30/></a>

[Lyria RealTime](https://deepmind.google/technologies/lyria/) は、最先端のリアルタイムストリーミング音楽生成モデルへのアクセスを提供します。開発者は、ユーザーがテキストプロンプトを使用してインタラクティブに楽器音楽を作成、継続的に操作、演奏できるアプリケーションを構築できます。

Lyria RealTime の主な特徴は次のとおりです。
* **最高品質のテキストから音声への変換モデル**: Lyria RealTime は、DeepMind が作成した最新のモデルを使用して、高品質のインストゥルメンタル音楽 (音声なし) を生成します。
* **ノンストップ音楽**: websocket を使用して、Lyria RealTime はリアルタイムで継続的に音楽を生成します。
* **影響の組み合わせ**: モデルにプロンプトを表示して、音楽のアイデア、ジャンル、楽器、ムード、または特徴を記述します。プロンプトを組み合わせて影響をブレンドし、ユニークな楽曲を作成できます。
* **クリエイティブコントロール**: `guidance`、`bpm`、音符/サウンドの `density`、`brightness`、`scale` をリアルタイムで設定します。モデルは新しい入力に基づいてスムーズに移行します。

詳細については、Lyria RealTime の[ドキュメント](https://ai.google.dev/gemini-api/docs/music-generation) を確認してください。

<!-- Notice Badge -->
<table align="left" border="3">
  <tr>
    <!-- Emoji -->
    <td bgcolor="#DCE2FF">
      <font size=30>🪧</font>
    </td>
    <!-- Text Content Cell -->
    <td bgcolor="#DCE2FF">
      <h4><font color=black>Lyria RealTime はプレビュー機能です。現在は割り当て制限付きで無料で使用できますが、変更される可能性があります。</font></h4>
    </td>
  </tr>
</table>

**また、Colab の制限により、Lyria RealTime のリアルタイム機能を体験することはできず、限られた音声出力しか得られないことにも注意してください。[Python スクリプト](./Get_started_LyriaRealTime.py) または AI Studio のアプリ、[Prompt DJ](https://aistudio.google.com/apps/bundled/promptdj) および [MIDI DJ](https://aistudio.google.com/apps/bundled/promptdj-midi) を使用して、Lyria RealTime を完全に体験してください**

# セットアップ

## SDK のインストール
このノートブックは SDK を使用しませんが、websocket と音声出力を管理するために python と colab 関数を使用します。


```
%pip install -U -q "google-genai>=1.16.0" # Lyria RealTime をサポートするには 1.16 が必要です
```

     [?25l    [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ [0m  [32m0.0/196.3 kB [0m  [31m? [0m eta  [36m-:--:-- [0m [2K    [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ [0m [91m╸ [0m  [32m194.6/196.3 kB [0m  [31m9.4 MB/s [0m eta  [36m0:00:01 [0m [2K    [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ [0m  [32m196.3/196.3 kB [0m  [31m4.3 MB/s [0m eta  [36m0:00:00 [0m
     [?25h

## API キー

次のセルを実行するには、API キーを `GOOGLE_API_KEY` という名前の Colab Secret に保存する必要があります。まだ API キーを持っていない場合、または Colab Secret の作成方法がわからない場合は、[認証](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) の例を参照してください。


```
from google.colab import userdata
import os

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
```

## モデルの選択と SDK クライアントの初期化

Lyria RealTime API は Lyria RealTime モデルで導入された新機能であるため、`lyria-realtime-exp` モデルでのみ動作します。

これは実験的な機能であるため、`v1alpha` クライアントバージョンを使用する必要もあります。



```
from google import genai
from google.genai import types

client = genai.Client(
    api_key=GOOGLE_API_KEY,
    http_options={'api_version': 'v1alpha'}, # Lyria RealTime は実験的なものであるため v1alpha
)

MODEL_ID = 'models/lyria-realtime-exp'
```

## ヘルパー


```
# @title Logging
# Lyria RealTime の仕組みを理解するために、すべてのログが表示されますが、
# 多すぎる場合はこれらの行を自由にコメントアウトしてください。

import logging

logger = logging.getLogger('Bidi')
logger.setLevel('DEBUG')
```


```
# @title Wave ファイルライター

import contextlib
import wave

@contextlib.contextmanager
def wave_file(filename, channels=2, rate=48000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        yield wf
```


```
# @title テキストからプロンプトへのパーサー

def parse_input(input_text):
  if ":" in input_text:
    parsed_prompts = []
    segments = input_text.split(',')
    malformed_segment_exists = False # いずれかのセグメントで解析エラーが発生したかどうかを追跡します

    for segment_str_raw in segments:
        segment_str = segment_str_raw.strip()
        if not segment_str: # 空のセグメントはスキップします (例: "text1:1, , text2:2" から)
            continue

        # プロンプトテキスト自体にコロンが含まれている場合に備えて、最初のコロンでのみ分割します
        parts = segment_str.split(':', 1)

        if len(parts) == 2:
            text_p = parts[0].strip()
            weight_s = parts[1].strip()

            if not text_p: # プロンプトテキストは空にしないでください
                print(f"エラー: セグメント '{segment_str_raw}' のプロンプトテキストが空です。このセグメントをスキップします。")
                malformed_segment_exists = True
                continue # この不正な形式のセグメントをスキップします
            try:
                weight_f = float(weight_s) # 重みは浮動小数点数です
                parsed_prompts.append(types.WeightedPrompt(text=text_p, weight=weight_f))
            except ValueError:
                print(f"エラー: セグメント '{segment_str_raw}' の重み '{weight_s}' が無効です。数値でなければなりません。このセグメントをスキップします。")
                malformed_segment_exists = True
                continue # この不正な形式のセグメントをスキップします
        else:
            # このセグメントは "text:weight" 形式ではありません。
            print(f"エラー: セグメント '{segment_str_raw}' は 'text:weight' 形式ではありません。このセグメントをスキップします。")
            malformed_segment_exists = True
            continue # この不正な形式のセグメントをスキップします

    if parsed_prompts: # 少なくとも 1 つのプロンプトが正常に解析された場合
        prompt_repr = [f"'{p.text}':{p.weight}" for p in parsed_prompts]
        if malformed_segment_exists:
            print(f"他のセグメントのエラーのため、{len(parsed_prompts)} 個の有効な重み付きプロンプトを部分的に送信しています: {', '.join(prompt_repr)}")
        else:
            print(f"複数の重み付きプロンプトを送信しています: {', '.join(prompt_repr)}")
        return parsed_prompts
    else: # ":" を含む入力文字列から有効なプロンプトが解析されなかった場合
        print("エラー: 入力に ':' が含まれていたため、複数プロンプト形式が示唆されましたが、有効な 'text:weight' セグメントが正常に解析されませんでした。アクションは実行されませんでした。")
        return None
  else:
    print(f"単一のテキストプロンプトを送信しています: \"{input_text}\" ")
    return types.WeightedPrompt(text=input_text, weight=1.0)

```

# メインオーディオループ

以下のクラスは、Lyria RealTime API との対話を実装します。

これは改善の余地のある基本的な実装ですが、理解しやすくするためにできるだけシンプルに保たれています。

[python スクリプト](Get_started_LyriaRealTime.py) は、より優れたスレッド処理とエラー処理、そして何よりもリアルタイムの対話を備えた、より完全な例です。

ここで説明する価値のある 2 つのメソッドがあります。

<h3><code>generate_music</code> - メイン関数</h3>

このメソッド:

- リアルタイム API に接続する `websocket` を開きます
- `session.set_weighted_prompts` を使用してモデルに初期プロンプトを送信します。何も指定されていない場合は、プロンプトを要求し、`parse_input` ヘルパーを使用して解析します。
- 指定されている場合は、`session.set_music_generation_config` を使用して音楽生成構成を送信します
- 最後に `session.play()` で音楽生成を開始します

<h3><code>receive</code> - API から音声データを収集して再生します</h3>

`receive` メソッドは、モデルの出力をリッスンし、ループで音声チャンクを収集し、`wave_file` ヘルパーを使用して `.wav` ファイルに書き込みます。特定の数のチャンク (デフォルトでは 10) の後に停止します。

Lyria RealTime とリアルタイムで対話したい場合は、新しいプロンプト/構成をモデルに送信するための `send` メソッドも実装する必要があります。このような例については、[python コードサンプル](./Get_started_LyriaRealTime.ipynb) を確認してください。


```python
import asyncio

file_index = 0

async def generate_music(prompts=None, max_chunks=10, config=None):
    async with client.aio.live.music.connect(model=MODEL_ID) as session:
        async def receive():
          global file_index
          # 新しい `.wav` ファイルを開始します。
          file_name = f"audio_{file_index}.wav"
          with wave_file(file_name) as wav:
            file_index += 1

            logger.debug('receive')

            # ソケットからチャンクを読み取ります。
            n = 0
            async for message in session.receive():
              n+=1
              if n > max_chunks:
                break

              # 音声チャンクを `.wav` ファイルに書き込みます。
              audio_chunk = message.server_content.audio_chunks[0].data
              if audio_chunk is not None:
                logger.debug('Got audio_chunk')
                wav.writeframes(audio_chunk)

              await asyncio.sleep(10**-12)

        # このコード例には、colab の制限によりリクエストを受信する方法がありません。
        # より完全な例については、python コードサンプルを確認してください

        while prompts is None:
          input_prompt = await asyncio.to_thread(input, "prompt > ")
          prompts = parse_input(input_prompt)

        # 指定されたプロンプトを送信します
        await session.set_weighted_prompts(
            prompts=prompts
        )

        # 初期構成を設定します
        if config is not None:
          await session.set_music_generation_config(config=config)

        # 音楽生成を開始します
        await session.play()

        receive_task = asyncio.create_task(receive())

        # タスクが完了するまでループを終了しません
        await asyncio.gather(receive_task)
```

# Lyria RealTime を試す

Colab の制限により、Lyria RealTime の「リアルタイム」部分を体験することはできないため、これらの例はすべて、音声ファイルを取得するための 1 回限りのプロンプトになります。

注意すべき点の 1 つは、音声は、すべてが wav ファイルに書き込まれたセッションの最後にのみ再生されることです。API を実際に使用すると、最初のチャンクが到着するとすぐに再生を開始できます。したがって、(専用のパラメーターを使用して) 設定した持続時間が長いほど、何か聞こえるまで待つ時間が長くなります。

## シンプルな Lyria RealTime の例
まず、簡単な例を示します。


```python
from IPython.display import display, Audio

await generate_music(prompts=[{"text":"piano", "weight":1.0}])
display(Audio(f"audio_{file_index-1}.wav"))
```

    DEBUG:Bidi:receive
    DEBUG:Bidi:Got audio_chunk
    DEBUG:Bidi:Got audio_chunk
    DEBUG:Bidi:Got audio_chunk
    DEBUG:Bidi:Got audio_chunk
    DEBUG:Bidi:Got audio_chunk
    DEBUG:Bidi:Got audio_chunk
    DEBUG:Bidi:Got audio_chunk
    DEBUG:Bidi:Got audio_chunk
    DEBUG:Bidi:Got audio_chunk
    DEBUG:Bidi:Got audio_chunk



<audio  controls="controls" >
    <source src="data:audio/x-wav;base64,UklGRiSYOgBXQVZFZm10IBAAAAABAAIAgLsAAADuAgAEABAAZGF0YQCYOgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQABAAEAAQABAAEAAgABAAMAAgAEAAIABAACAAUAAgAFAAIABgADAAYAAwAHAAMABwADAAgAAwAJAAMACgACAAoAAgAKAAMACwADAAwABAANAAQADAADAAwAAgALAAEADAAAAA0A//8NAP3/DQD8/w0A/P8NAPz/DQD8/w0A/f8OAP//DwABAA8AAQAOAAEADQAEAA4ACQAQAA0AEgARABQAFAAVABUAFQAUABMAEgASAA8AEgAPABIAEAASABEAEgASABIAEgATABIAEwARABMADgAQAA0ADwAOABIAEQAUABMAEwAWABEAFwAQABkAEAAbABAAHQAPAB8AEAAfABIAHQASABsAEAAYAAwAFwAIABYABgAXAAUAFwAGABUABQAUAAQAFQAFABcABwAbAAgAIAALACMADQAmAA4AJwANACcADAAlAAsAIgAKAB8ACQAcAAYAFwAAABIA+f8PAPD/DQDp/wwA5/8KAOb/BgDl/wIA5P/+/+f//v/p/wEA5/8EAOT/BADk/wAA5P/9/+b/+//n//j/5//0/+b/8//l//L/5v/v/+f/6f/p/+T/6P/i/+b/4v/j/+H/3//d/9r/1//T/9H/zP/M/8j/zv/G/9T/wv/W/7z/0f+5/8z/uv/J/7v/xv+7/8L/u/++/7//uf/G/7T/zf+x/9H/rv/U/63/1v+u/9n/sv/a/7f/2f+6/9r/u//d/7//4P/E/93/xv/W/8b/0P/E/83/wP/M/7n/zP+0/8n/s//H/7P/xP+y/8H/sf++/7L/wP+2/8T/uv/H/7z/yP++/8j/wP/H/8X/yP/K/9H/z//g/9P/7P/W//L/2//5/97//f/f////3/8DAOL/BwDj/wYA4f8DAN///f/e//T/3//n/+P/4P/r/97/+P/f/wIA3v8HAOH/CQDl/wsA5v8RAOH/FADZ/xUA0f8TAMr/DwDH/wwAxf8MAMX/DwDE/xQAxP8dAMf/KADQ/zEA2/83AOP/QQDm/1AA5f9ZAOL/XADe/18A2P9dAM7/VQDK/04AzP9OAMr/UgDG/1cAxf9eAMr/ZADP/2gA0v9nANL/YgDV/2QA3f9sAOH/bwDd/20A1P9xAM7/fwDM/4kAx/+EAL//eAC8/28AwP9rAMX/ZgDF/2AAwP9cALz/WQC+/1YAw/9LAMn/OgDL/yoAyf8nAMP/KgC//y0Avf8wAL7/MwDB/zMAxP8tAMT/IwDB/xwAvf8bALv/IQC+/ykAzf8tAN7/MADp/zMA7v82APT/OAD6/zkA/f8+AP//RgAFAE8ACgBTAA0ATgAQAEgAEgBJABAAUQAQAFkAFQBdAB4AXQAqAGMAOgB0AEsAigBYAJgAXQCbAF4AkgBiAIEAaQB2AG8AewB3AIcAfACLAHkAiwB4AIcAfgB+AIIAdQCDAHMAhgB4AJEAfwCiAIIAtQCDAMkAiwDVAJkA2gCmANsAqQDSAKUAxACiAMEApwDJAKkAzAChAMIAlQC4AIwAswCHALUAiAC8AIoAvwCHAL0AggC4AH4AtQB/ALUAfgC3AHcAugBvAL0AawC9AGYAtQBgAKgAXQCcAF4AlgBdAJQAUwCQAEYAjAA8AIYANAB+ACwAcgAeAGIADgBSAAIASgD9/0YA9v87AO3/LADp/x8A6v8XAOv/FQDq/yEA7f82APf/RAD//0EAAQA1AAAAKQD//yAA/P8aAPn/FgD0/xAA5f8JAND/AQDC//X/wf/o/87/5//d//D/5v/y/+j/7P/r/+n/8v/t//T/6//x/9z/8P/I/+//uv/v/7b/8/+4//X/u//z/7v/8/+0//j/p/8BAKD/BwCj/wcAqP8DAK//BgC5/wwAxv8NANP/DQDc/xM... [truncated]
    お使いのブラウザは audio 要素をサポートしていません。
</audio>


# 次のステップ

音楽の生成方法がわかったので、次に試すべきクールなことをいくつか紹介します。
*   音楽の代わりに、[TTS モデル](./Get_started_TTS.ipynb) を使用して複数の話者の会話を生成する方法を学びます。
*   [画像](./Get_started_Imagen.ipynb) または [動画](./Get_started_Veo.ipynb) を生成する方法を発見します。
*   音楽や音声を生成する代わりに、Gemini が [音声ファイル](./Audio.ipynb) を理解する方法を学びます。
*   [Live API](./Get_started_LiveAPI.ipynb) を使用して Gemini とリアルタイムで会話します。

```