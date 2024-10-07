![AutoRes_logo](images/AutoRes_logo.png "AutoRes_logo")

# ResearchGraph
ResearchGraphはAI研究の目的を与えるだけで、AIが完全自律的に調査・研究を行い、新たな手法の作成・検証を行うシステムです🤖。 
ResearchGraphはLangGraph(https://www.langchain.com/langgraph)の記載方法に則り設計されています。pipからライブラリをインストールするだけで使いことができます！  
ResearchGraphの特徴としては以下のものが挙げられます。
1. 自動研究によるサイクル  
   研究の全プロセスを自動化することができます。また研究サイクルを研究対象とすることによってシステムの最適化も行うことができます。
2. 仮想空間で閉じた学習  
   システムとして仮想空間内に閉じることによって、人間が介入しなくても良くなり、結果的に研究速度が向上します。

<p align="left">
  📖 <a href="https://arxiv.org/abs/2408.06292">[論文]</a> |
  💻 <a href="https://www.autores.one">[プロジェクトサイト]</a> |
</p>

![ResearchGraphのアーキテクチャ図](images/research_graph.png "ResearchGraph")
![ResearchGraphのイラスト図](images/images.png "ResearchGraph")


## 目次

1. [インストール要件](#インストール要件)
2. [インストール方法](#インストール方法)
3. [利用ガイド](#利用ガイド)
4. [カスタマイズ](#カスタマイズ)
5. [サンプル](#サンプル)
6. [ライセンス](#ライセンス)
7. [引用方法](#引用方法)


## インストール要件

・Python >=3.10
・動作確認済みLLM API
| API | モデル |
| ---- | ---- |
| OpenAI API| gpt-4o-2024-08-06<br> gpt-4o-2024-05-13<br> gpt-4o-mini-2024-07-18<br> gpt-4-turbo-2024-04-09<br> gpt-4-0125-preview|
| ANTHROPIC API | claude-3-5-sonnet-20240620<br> claude-3-opus-20240229|
| Google API | gemini-1.0-pro<br> gemini-1.5-pro<br> gemini-1.5-flash|


## インストール方法

最新のResearchGraphはpip経由でインストールすることができます。これによって新たに追加されたノードを活用することができます。
```bash
pip install --upgrade -q ResearchGraph
```

## 利用ガイド
LangGraphに則り、利用ガイドとして以下の内容に関してサンプルノートブックに記載しています。
・Nodeの組み合わせ方法
・Graphの実行方法 
### Nodeの組み合わせ方法
(ここにジュビターノートブックのURL)

### Graphの実行方法 
(ここにジュビターノートブックのURL)

## カスタマイズ
ResearchGraphのノードなどをカスタマイズで作成する方法など記載しますか？？

## サンプル
具体的な例を用いたResearchGraphの使用方法をGoogleColabNotebookに示します。  
1. 最適化手法の研究  
   ・目的として既存の最適化手法における計算コストの低減に向けた研究を実行します
2. ・・・

## ライセンス
ResearchGraphは〇〇ライセンスが適用されています。

## 引用方法
```
@inproceedings{shen2024variational,
      title={Variational Learning is Effective for Large Deep Networks}, 
      author={Yuesong Shen and Nico Daheim and Bai Cong and Peter Nickl and Gian Maria Marconi and Clement Bazan and Rio Yokota and Iryna Gurevych and Daniel Cremers and Mohammad Emtiyaz Khan and Thomas Möllenhoff},
      booktitle={International Conference on Machine Learning (ICML)},
      year={2024},
      url={https://arxiv.org/abs/2402.17641}
}
```


