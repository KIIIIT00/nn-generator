# No-Code Neural Network Code Generator
[![CodeQL Advanced](https://github.com/KIIIIT00/nn-generator/actions/workflows/codeql.yml/badge.svg)](https://github.com/KIIIIT00/nn-generator/actions/workflows/codeql.yml)
<img src="https://img.shields.io/badge/Javascript-276DC3.svg?logo=javascript&style=flat">
<img src="https://img.shields.io/badge/-React-555.svg?logo=react&style=flat">
<img src="https://img.shields.io/badge/-Python-F9DC3E.svg?logo=python&style=flat">
![NN Generator](/images/nn-generator.png)
## About
Neural Network Code Generatorは，PyTorchベースのニューラルネットワークを構築するためのPythonコードを簡単に作成できるWebアプリケーションです．
コーディングの知識がなくても，UIを使用して直感的にニューラルネットワークを設計し，すぐに使えるコードを生成できる．
デモは，[こちら](https://kiiiit00.github.io/nn-generator/)です．

## Features
- **ノーコード設計**：プログラミング知識が不要でニューラルネットワークのコードを生成
- **カスタマイズ可能**:レイヤー，ニューロン数，活性化関数などを自由に設定
- **豊富なオプション**:様々なオプティマイザー，損失関数，データセットをサポート
- **可視化コード**:トレーニング過程の可視化コードも生成可能
- **二言語対応**:英語と日本語の切り替え可能
- **コード出力**:生成したコードをコピーまたはファイルとしてダウンロード可能

## Setup
### Requirements
- Node.js(v12.0.0以上)
- npm(v6.0.0以上)

### Installation
```
# clone the repocitory
$ git clone https://github.com/KIIIIT00/nn-generator.git
$ cd nn-generator

## Install dependencies
$ npm install

## Start the application
$ npm start
```
ブラウザで http://localhost:3000 を開くと、アプリケーションが表示されます．
## Usage
1. データセット設定：使用するデータセットを選択（MNIST, Fashion MNIST, CIFAR-10, カスタム）
2. トレーニング設定：オプティマイザー，学習率，損失関数，エポック数，バッチサイズを設定
3. レイヤー設定：各レイヤーのニューロン数と活性化関数を設定（追加・削除も可能）
4. 「Generate Code」ボタンをクリックしてコードを作成
5. 生成されたコードはコピーまたはダウンロード可能

## Techonologies
- **フロントエンド**：React, TailwindCSS
- **生成コード**: Python（Pytorch）
- **デプロイ**:GitHub Pages
## Project Structure
```
neural-network-generator/
├── public/                  
├── src/                     # ソースコード
│   ├── components/          # Reactコンポネント
│   │   ├── NNGenerator.jsx  
│   │   ├── CodeTemplates.js 
│   │   └── translation.js  
│   ├── App.js               
│   └── index.js             
└── package.json             
```
## License
このプロジェクトはMITライセンスの下で公開されています．詳しくはLICENSEファイルを参照してください

## Contributing
バグ報告や機能リクエストは，GitHubのIssuesに投稿してください．

---

### Note:
生成されたコードを使用する場合は，PyTorchとその依存関係をインストールする必要があります．
```
$ pip install torch torchvision matplotlib numpy
```

