# ResearchChain
ResearchChainはAI研究の目的を与えるだけで、AIが完全自律的に調査・研究を行い、新たな手法の作成・検証を行うシステムです。
pipからライブラリをインストールするだけで使いことができます！

## 目次

1. [インストール要件](#インストール要件)
2. [インストール方法](#インストール方法)
3. [利用ガイド](#利用ガイド)


## インストール要件

・Python >=3.10

## インストール方法

最新のResearchChainはpip経由でインストールすることができます。これによって新たに追加されたノードを活用することができます。
```bash
pip install --upgrade -q researchchain
```

## 利用ガイド

論文中で実施した実験を例に、利用方法を記載します。

### LLM APIキーの設定

ResearchChainで用いるLLMのAPIキーを以下のコードで指定します。

```bash
import os
os.environ["OPENAI_API_KEY"] = "ここに用いるAPIキーを入力"
```

ResearchChainは様々なLLMに対応しており、利用者が好きなLLMを選択して利用することができます。ただし研究ではGPT-4oを活用しているため、それ以降のモデルの活用を推奨します。
サポートしているLLMの一覧
| LLM提供 | モデル |
| ---- | ---- |
| OpenAI | GPT-4o<br> GPT-4<br> GPT-3.5|
| Claude | Claude-3.5-sonnet|

### 


