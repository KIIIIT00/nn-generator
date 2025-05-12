import React from 'react';
import { useState, useEffect } from 'react';

const NNGen = () => {
  // 初期設定値
  const [layers, setLayers] = useState([
    { neurons: 64, activation: 'relu' },
    { neurons: 32, activation: 'relu' },
    { neurons: 1, activation: 'sigmoid' }
  ]);
  const [optimizer, setOptimizer] = useState('adam');
  const [loss, setLoss] = useState('bce');
  const [epochs, setEpochs] = useState(10);
  const [batchSize, setBatchSize] = useState(32);
  const [learningRate, setLearningRate] = useState(0.001);
  const [dataset, setDataset] = useState('custom');
  const [includeVisualization, setIncludeVisualization] = useState(true);
  const [generatedCode, setGeneratedCode] = useState('');
  const [inputShape, setInputShape] = useState(10);

  // 活性化関数オプション
  const activationOptions = [
    'relu', 'sigmoid', 'tanh', 'softmax', 'leaky_relu', 'elu', 'selu', 'gelu', 'prelu', 'none'
  ];

  // オプティマイザーオプション
  const optimizerOptions = [
    'adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamw'
  ];

  // 損失関数オプション
  const lossOptions = [
    'bce', 'bce_with_logits', 'cross_entropy', 'nll', 'mse', 'l1', 'smooth_l1', 'huber'
  ];

  // データセットオプション
  const datasetOptions = [
    'mnist', 'fashion_mnist', 'cifar10', 'custom'
  ];

  // レイヤーを追加
  const addLayer = () => {
    setLayers([...layers, { neurons: 32, activation: 'relu' }]);
  };

  // レイヤーを削除
  const removeLayer = (index) => {
    if (layers.length > 1) {
      const newLayers = [...layers];
      newLayers.splice(index, 1);
      setLayers(newLayers);
    }
  };

  // レイヤーを更新
  const updateLayer = (index, field, value) => {
    const newLayers = [...layers];
    newLayers[index] = { ...newLayers[index], [field]: value };
    setLayers(newLayers);
  };

  // コードをコピー
  const copyCode = () => {
    navigator.clipboard.writeText(generatedCode)
      .then(() => {
        alert('コードがクリップボードにコピーされました！');
      })
      .catch((err) => {
        console.error('コピーに失敗しました:', err);
        alert('コピーに失敗しました。手動でコピーしてください。');
      });
  };

  // コードをダウンロード
  const downloadCode = () => {
    const element = document.createElement('a');
    const file = new Blob([generatedCode], {type: 'text/plain'});
    element.href = URL.createObjectURL(file);
    element.download = 'neural_network.py';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  // インデントを削除して左詰めにするヘルパー関数
  const removeIndentation = (code) => {
    // コードの各行を取得
    const lines = code.split('\n');
    
    // 空でない行の中で最小のインデントを見つける
    let minIndent = Infinity;
    lines.forEach(line => {
      if (line.trim() !== '') {
        const indentMatch = line.match(/^\s*/);
        if (indentMatch && indentMatch[0].length < minIndent) {
          minIndent = indentMatch[0].length;
        }
      }
    });
    
    // 最小インデントが無限の場合（全ての行が空）は0に設定
    if (minIndent === Infinity) {
      minIndent = 0;
    }
    
    // 各行からインデントを削除
    return lines.map(line => {
      if (line.trim() === '') {
        return '';
      }
      return line.substring(minIndent);
    }).join('\n');
  };

  // コードを生成
  const generateCode = () => {
    let code = `import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
`;

    // データセットコード
    if (dataset !== 'custom') {
      code += `
# データセットの読み込み
import torchvision
import torchvision.transforms as transforms
`;
      if (dataset === 'mnist') {
        code += `
# MNISTデータセットの読み込み
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# データローダーの作成
train_loader = DataLoader(train_dataset, batch_size=${batchSize}, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=${batchSize}, shuffle=False)

# 入力形状の設定
input_size = 784  # 28x28
`;
      } else if (dataset === 'fashion_mnist') {
        code += `
# Fashion MNISTデータセットの読み込み
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# データローダーの作成
train_loader = DataLoader(train_dataset, batch_size=${batchSize}, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=${batchSize}, shuffle=False)

# 入力形状の設定
input_size = 784  # 28x28
`;
      } else if (dataset === 'cifar10') {
        code += `
# CIFAR-10データセットの読み込み
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# データローダーの作成
train_loader = DataLoader(train_dataset, batch_size=${batchSize}, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=${batchSize}, shuffle=False)

# 入力形状の設定
input_size = 3 * 32 * 32  # 3x32x32
`;
      }
    } else {
      code += `
# カスタムデータセット
# ここにデータ読み込みと前処理コードを追加
X = torch.randn(1200, ${inputShape})
y = torch.randint(0, 2, (1200, 1)).float()

# データセットを訓練用と検証用に分割
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# データローダーの作成
train_loader = DataLoader(train_dataset, batch_size=${batchSize}, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=${batchSize}, shuffle=False)

# 入力形状の設定
input_size = ${inputShape}
`;
    }

    // モデルクラスの作成
    code += `
# ニューラルネットワークモデルの定義
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
`;

    // レイヤーの追加
    let previousSize = (dataset === 'mnist' || dataset === 'fashion_mnist') ? 784 : 
                      (dataset === 'cifar10') ? 3 * 32 * 32 : inputShape;
    
    layers.forEach((layer, index) => {
      let activation = '';
      if (layer.activation === 'relu') {
        activation = 'nn.ReLU()';
      } else if (layer.activation === 'sigmoid') {
        activation = 'nn.Sigmoid()';
      } else if (layer.activation === 'tanh') {
        activation = 'nn.Tanh()';
      } else if (layer.activation === 'softmax') {
        activation = 'nn.Softmax(dim=1)';
      } else if (layer.activation === 'leaky_relu') {
        activation = 'nn.LeakyReLU(0.1)';
      } else if (layer.activation === 'elu') {
        activation = 'nn.ELU()';
      } else if (layer.activation === 'selu') {
        activation = 'nn.SELU()';
      } else if (layer.activation === 'gelu') {
        activation = 'nn.GELU()';
      } else if (layer.activation === 'prelu') {
        activation = 'nn.PReLU()';
      } else if (layer.activation === 'none') {
        activation = '';
      }
      
      code += `            nn.Linear(${previousSize}, ${layer.neurons}),\n`;
      if (activation) {
        code += `            ${activation},\n`;
      }
      
      previousSize = layer.neurons;
    });
    
    code = code.slice(0, -2); // 最後のカンマと改行を削除
    
    code += `
        )
    
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # フラット化
        return self.layers(x)

# モデルとデバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork().to(device)
print(model)

# 損失関数とオプティマイザーの設定
`;

    // 損失関数の設定
    if (loss === 'bce') {
      code += `criterion = nn.BCELoss()\n`;
    } else if (loss === 'bce_with_logits') {
      code += `criterion = nn.BCEWithLogitsLoss()\n`;
    } else if (loss === 'cross_entropy') {
      code += `criterion = nn.CrossEntropyLoss()\n`;
    } else if (loss === 'nll') {
      code += `criterion = nn.NLLLoss()\n`;
    } else if (loss === 'mse') {
      code += `criterion = nn.MSELoss()\n`;
    } else if (loss === 'l1') {
      code += `criterion = nn.L1Loss()\n`;
    } else if (loss === 'smooth_l1') {
      code += `criterion = nn.SmoothL1Loss()\n`;
    } else if (loss === 'huber') {
      code += `criterion = nn.HuberLoss()\n`;
    }

    // オプティマイザーの設定
    if (optimizer === 'adam') {
      code += `optimizer = optim.Adam(model.parameters(), lr=${learningRate})\n`;
    } else if (optimizer === 'sgd') {
      code += `optimizer = optim.SGD(model.parameters(), lr=${learningRate}, momentum=0.9)\n`;
    } else if (optimizer === 'rmsprop') {
      code += `optimizer = optim.RMSprop(model.parameters(), lr=${learningRate})\n`;
    } else if (optimizer === 'adagrad') {
      code += `optimizer = optim.Adagrad(model.parameters(), lr=${learningRate})\n`;
    } else if (optimizer === 'adadelta') {
      code += `optimizer = optim.Adadelta(model.parameters(), lr=${learningRate})\n`;
    } else if (optimizer === 'adamw') {
      code += `optimizer = optim.AdamW(model.parameters(), lr=${learningRate})\n`;
    }

    // 訓練コード
    code += `
# 訓練関数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 勾配をゼロにリセット
        optimizer.zero_grad()
        
        # 順伝播 + 逆伝播 + 最適化
        outputs = model(inputs)
        
        # ターゲットの形状を調整（必要な場合）
        if outputs.shape != targets.shape:
            if outputs.shape[1] == 1:
                # バイナリ分類の場合
                targets = targets.float().view(-1, 1)
            else:
                # 多クラス分類の場合、ラベルとしての形状に変換
                targets = targets.long().view(-1)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # 精度の計算（バイナリ分類またはマルチクラス分類に対応）
        if outputs.shape[1] == 1:  # バイナリ分類
            predicted = (outputs > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        else:  # マルチクラス分類
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# 評価関数
def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            
            # ターゲットの形状を調整（必要な場合）
            if outputs.shape != targets.shape:
                if outputs.shape[1] == 1:
                    # バイナリ分類の場合
                    targets = targets.float().view(-1, 1)
                else:
                    # 多クラス分類の場合、ラベルとしての形状に変換
                    targets = targets.long().view(-1)
            
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            
            # 精度の計算（バイナリ分類またはマルチクラス分類に対応）
            if outputs.shape[1] == 1:  # バイナリ分類
                predicted = (outputs > 0.5).float()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            else:  # マルチクラス分類
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# 訓練ループ
train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(${epochs}):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

print('訓練完了')
`;

    // 可視化コード
    if (includeVisualization) {
      code += `
# 訓練過程の可視化
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_accs)
plt.plot(val_accs)
plt.title('モデル精度')
plt.ylabel('精度 (%)')
plt.xlabel('エポック')
plt.legend(['訓練', '検証'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(train_losses)
plt.plot(val_losses)
plt.title('モデル損失')
plt.ylabel('損失')
plt.xlabel('エポック')
plt.legend(['訓練', '検証'], loc='upper right')

plt.tight_layout()
plt.show()
`;
    }

    // モデル保存コード
    code += `
# モデルの保存
torch.save(model.state_dict(), 'model.pth')
print("モデルを保存しました。")
`;

    // インデントを削除して左詰めにする
    const formattedCode = removeIndentation(code);
    setGeneratedCode(formattedCode);
  };

  // コンポーネントがマウントされたときに最初のコード生成を実行
  useEffect(() => {
    generateCode();
  }, []);

  return (
    <div className="flex flex-col p-4 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-6 text-center">ノーコード ニューラルネットワーク コードジェネレーター</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <div className="bg-gray-100 p-4 rounded-lg">
          <h2 className="text-lg font-semibold mb-2">データセット設定</h2>
          <div className="mb-2">
            <label className="block text-sm mb-1">データセット:</label>
            <select
              className="w-full p-2 border rounded"
              value={dataset}
              onChange={(e) => setDataset(e.target.value)}
            >
              {datasetOptions.map(option => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </div>
          
          {dataset === 'custom' && (
            <div className="mb-2">
              <label className="block text-sm mb-1">入力形状 (特徴量の数):</label>
              <input
                type="number"
                className="w-full p-2 border rounded"
                value={inputShape}
                onChange={(e) => setInputShape(parseInt(e.target.value))}
                min="1"
              />
            </div>
          )}
        </div>
        
        <div className="bg-gray-100 p-4 rounded-lg">
          <h2 className="text-lg font-semibold mb-2">トレーニング設定</h2>
          <div className="mb-2">
            <label className="block text-sm mb-1">オプティマイザー:</label>
            <select
              className="w-full p-2 border rounded"
              value={optimizer}
              onChange={(e) => setOptimizer(e.target.value)}
            >
              {optimizerOptions.map(option => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </div>
          
          <div className="mb-2">
            <label className="block text-sm mb-1">学習率:</label>
            <input
              type="number"
              className="w-full p-2 border rounded"
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value))}
              min="0.0001"
              max="1"
              step="0.0001"
            />
          </div>
          
          <div className="mb-2">
            <label className="block text-sm mb-1">損失関数:</label>
            <select
              className="w-full p-2 border rounded"
              value={loss}
              onChange={(e) => setLoss(e.target.value)}
            >
              {lossOptions.map(option => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </div>
          
          <div className="mb-2">
            <label className="block text-sm mb-1">エポック数:</label>
            <input
              type="number"
              className="w-full p-2 border rounded"
              value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value))}
              min="1"
            />
          </div>
          
          <div className="mb-2">
            <label className="block text-sm mb-1">バッチサイズ:</label>
            <input
              type="number"
              className="w-full p-2 border rounded"
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value))}
              min="1"
            />
          </div>
        </div>
        
        <div className="bg-gray-100 p-4 rounded-lg">
          <h2 className="text-lg font-semibold mb-2">追加設定</h2>
          <div className="mb-4">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={includeVisualization}
                onChange={(e) => setIncludeVisualization(e.target.checked)}
                className="mr-2"
              />
              <span>訓練過程の可視化を含める</span>
            </label>
          </div>
        </div>
      </div>
      
      <div className="bg-gray-100 p-4 rounded-lg mb-4">
        <div className="flex justify-between items-center mb-2">
          <h2 className="text-lg font-semibold">レイヤー設定</h2>
          <button 
            onClick={addLayer}
            className="bg-blue-500 text-white px-3 py-1 rounded hover:bg-blue-600"
          >
            レイヤーを追加
          </button>
        </div>
        
        {layers.map((layer, index) => (
          <div key={index} className="flex flex-wrap items-center mb-2 p-2 border rounded bg-white">
            <div className="mr-2 mb-2 md:mb-0">
              <span className="font-medium">レイヤー {index + 1}:</span>
            </div>
            
            <div className="mr-2 mb-2 md:mb-0">
              <label className="text-sm mr-1">ニューロン数:</label>
              <input
                type="number"
                className="w-20 p-1 border rounded"
                value={layer.neurons}
                onChange={(e) => updateLayer(index, 'neurons', parseInt(e.target.value))}
                min="1"
              />
            </div>
            
            <div className="mr-2 mb-2 md:mb-0">
              <label className="text-sm mr-1">活性化関数:</label>
              <select
                className="p-1 border rounded"
                value={layer.activation}
                onChange={(e) => updateLayer(index, 'activation', e.target.value)}
              >
                {activationOptions.map(option => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
            </div>
            
            <div className="ml-auto">
              {layers.length > 1 && (
                <button
                  onClick={() => removeLayer(index)}
                  className="bg-red-500 text-white px-2 py-1 rounded hover:bg-red-600 text-sm"
                >
                  削除
                </button>
              )}
            </div>
          </div>
        ))}
      </div>
      
      <div className="mb-4">
        <button
          onClick={generateCode}
          className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 w-full font-bold"
        >
          Pythonコードを生成
        </button>
      </div>
      
      <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-auto">
        <div className="flex justify-between items-center mb-2">
          <h2 className="text-lg font-semibold">生成されたコード</h2>
          <div className="flex">
            <button
              onClick={copyCode}
              className="bg-blue-500 text-white px-3 py-1 rounded hover:bg-blue-600 mr-2"
            >
              コピー
            </button>
            <button
              onClick={downloadCode}
              className="bg-green-500 text-white px-3 py-1 rounded hover:bg-green-600"
            >
              ダウンロード
            </button>
          </div>
        </div>
        <pre className="whitespace-pre text-left overflow-x-auto">
          <code>{generatedCode}</code>
        </pre>
      </div>
    </div>
  );
};

export default NNGen;