import React from 'react';
import { useState, useEffect } from 'react';

import {
  ImportCode,
  DataCode,
  ModelCode,
  TrainingCode,
  VisualizationCode,
  SaveModelCode
} from './CodeTemplate';

const NNGen = () => {
  // Initial settings
  const [layers, setLayers] = useState([
    { neurons: 64, activation: 'relu' },
    { neurons: 32, activation: 'relu' },
    { neurons: 1, activation: 'sigmoid' }
  ]);
  const [optimizer, setOptimizer] = useState('adam');
  const [loss, setLoss] = useState('bce');
  const [epochs, setEpochs] = useState(10);
  const [batchSize, setBatchSize] = useState(32);
  const [lr, setLr] = useState(0.001);
  const [dataset, setDataset] = useState('custom');
  const [includeVisualization, setIncludeVisualization] = useState(true);
  const [generatedCode, setGeneratedCode] = useState('');
  const [inputShape, setInputShape] = useState(10);

  // Activation function options
  const activationOptions = [
    'relu', 'sigmoid', 'tanh', 'softmax', 'leaky_relu', 'elu', 'selu', 'gelu', 'prelu', 'none'
  ];

  // Optimizer options
  const optimizerOptions = [
    'adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamw'
  ];

  // Loss function options
  const lossOptions = [
    'bce', 'bce_with_logits', 'cross_entropy', 'nll', 'mse', 'l1', 'smooth_l1', 'huber'
  ];

  // Dataset options
  const datasetOptions = [
    'mnist', 'fashion_mnist', 'cifar10', 'custom'
  ];

  // Add a layer
  const addLayer = () => {
    setLayers([...layers, { neurons: 32, activation: 'relu' }]);
  };

  // Remove a layer
  const removeLayer = (index) => {
    if (layers.length > 1) {
      const newLayers = [...layers];
      newLayers.splice(index, 1);
      setLayers(newLayers);
    }
  };

  // Update a layer
  const updateLayer = (index, field, value) => {
    const newLayers = [...layers];
    newLayers[index] = { ...newLayers[index], [field]: value };
    setLayers(newLayers);
  };

  // Copy code
  const copyCode = () => {
    navigator.clipboard.writeText(generatedCode)
      .then(() => {
        alert('Code copied to clipboard!');
      })
      .catch((err) => {
        console.error('Copy failed:', err);
        alert('Copy failed. Please copy manually.');
      });
  };

  // Download code
  const downloadCode = () => {
    const element = document.createElement('a');
    const file = new Blob([generatedCode], {type: 'text/plain'});
    element.href = URL.createObjectURL(file);
    element.download = 'neural_network.py';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  // Helper function to remove indentation and left-align code
  const removeIndentation = (code) => {
    // Get each line of code
    const lines = code.split('\n');
    
    // Find the minimum indentation among non-empty lines
    let minIndent = Infinity;
    lines.forEach(line => {
      if (line.trim() !== '') {
        const indentMatch = line.match(/^\s*/);
        if (indentMatch && indentMatch[0].length < minIndent) {
          minIndent = indentMatch[0].length;
        }
      }
    });
    
    // If minimum indent is infinity (all lines are empty), set to 0
    if (minIndent === Infinity) {
      minIndent = 0;
    }
    
    // Remove indentation from each line
    return lines.map(line => {
      if (line.trim() === '') {
        return '';
      }
      return line.substring(minIndent);
    }).join('\n');
  };

  // Generate code
  const generateCode = () => {
    const config = {
      layers,
      optimizer,
      loss,
      epochs,
      batchSize,
      lr,
      dataset,
      includeVisualization,
      inputShape
    };

    let code = ImportCode(config);
    code += DataCode(config);
    code += ModelCode(config);
    code += TrainingCode(config);
    if (includeVisualization) {
      code += VisualizationCode(config);
    }
    code += SaveModelCode(config);
    // Remove indentation and left-align
    const formattedCode = removeIndentation(code);
    setGeneratedCode(formattedCode);
  };

  // Execute initial code generation when component mounts
  useEffect(() => {
    generateCode();
  }, []);

  return (
    <div className="flex flex-col p-4 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-6 text-center">No-Code Neural Network Code Generator</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <div className="bg-gray-100 p-4 rounded-lg">
          <h2 className="text-lg font-semibold mb-2">Dataset Settings</h2>
          <div className="mb-2">
            <label className="block text-sm mb-1">Dataset:</label>
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
              <label className="block text-sm mb-1">Input Shape (number of features):</label>
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
          <h2 className="text-lg font-semibold mb-2">Training Settings</h2>
          <div className="mb-2">
            <label className="block text-sm mb-1">Optimizer:</label>
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
            <label className="block text-sm mb-1">Learning Rate:</label>
            <input
              type="number"
              className="w-full p-2 border rounded"
              value={lr}
              onChange={(e) => setLr(parseFloat(e.target.value))}
              min="0.0001"
              max="1"
              step="0.0001"
            />
          </div>
          
          <div className="mb-2">
            <label className="block text-sm mb-1">Loss Function:</label>
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
            <label className="block text-sm mb-1">Epochs:</label>
            <input
              type="number"
              className="w-full p-2 border rounded"
              value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value))}
              min="1"
            />
          </div>
          
          <div className="mb-2">
            <label className="block text-sm mb-1">Batch Size:</label>
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
          <h2 className="text-lg font-semibold mb-2">Additional Settings</h2>
          <div className="mb-4">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={includeVisualization}
                onChange={(e) => setIncludeVisualization(e.target.checked)}
                className="mr-2"
              />
              <span>Include training visualization</span>
            </label>
          </div>
        </div>
      </div>
      
      <div className="bg-gray-100 p-4 rounded-lg mb-4">
        <div className="flex justify-between items-center mb-2">
          <h2 className="text-lg font-semibold">Layer Settings</h2>
          <button 
            onClick={addLayer}
            className="bg-blue-500 text-white px-3 py-1 rounded hover:bg-blue-600"
          >
            Add Layer
          </button>
        </div>
        
        {layers.map((layer, index) => (
          <div key={index} className="flex flex-wrap items-center mb-2 p-2 border rounded bg-white">
            <div className="mr-2 mb-2 md:mb-0">
              <span className="font-medium">Layer {index + 1}:</span>
            </div>
            
            <div className="mr-2 mb-2 md:mb-0">
              <label className="text-sm mr-1">Neurons:</label>
              <input
                type="number"
                className="w-20 p-1 border rounded"
                value={layer.neurons}
                onChange={(e) => updateLayer(index, 'neurons', parseInt(e.target.value))}
                min="1"
              />
            </div>
            
            <div className="mr-2 mb-2 md:mb-0">
              <label className="text-sm mr-1">Activation Function:</label>
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
                  Delete
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
          Generate Python Code
        </button>
      </div>
      
      <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-auto">
        <div className="flex justify-between items-center mb-2">
          <h2 className="text-lg font-semibold">Generated Code</h2>
          <div className="flex">
            <button
              onClick={copyCode}
              className="bg-blue-500 text-white px-3 py-1 rounded hover:bg-blue-600 mr-2"
            >
              Copy
            </button>
            <button
              onClick={downloadCode}
              className="bg-green-500 text-white px-3 py-1 rounded hover:bg-green-600"
            >
              Download
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