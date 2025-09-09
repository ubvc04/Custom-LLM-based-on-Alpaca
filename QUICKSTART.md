# 🚀 Quick Start Guide - Alpaca Domination

## 🎯 Project Overview

Welcome to **Alpaca Domination** - the most ambitious project to create the world's #1 Alpaca-trained language model! This guide will get you up and running quickly.

## ⚡ Quick Setup (5 minutes)

### 1. Download the Dataset

```bash
# Run this to download and prepare the Alpaca dataset
python scripts/simple_download.py
```

### 2. Install Dependencies

```bash
# Activate virtual environment
alpaca_llm_env\Scripts\activate  # Windows
# or
source alpaca_llm_env/bin/activate  # Linux/Mac

# Install core packages
pip install torch transformers datasets pandas numpy rich fastapi uvicorn
```

### 3. Start the Backend API

```bash
cd backend
python main.py
```

The API will be available at `http://localhost:8000`

### 4. Start the Frontend (Optional)

```bash
cd frontend
npm install
npm run dev
```

The web interface will be available at `http://localhost:3000`

## 🎯 Training the Model

### Quick Training (Demo)

```bash
# Basic training run
python scripts/train_model.py
```

### Advanced Training

```bash
# With custom configuration
python scripts/train_model.py --config config/alpaca_7b.yaml
```

## 📊 Project Structure

```
alpaca-domination/
├── 🧠 src/models/           # Model architectures
│   └── alpaca_domination.py # Main model implementation
├── 📊 data/                 # Training data
├── 🚀 scripts/              # Training & setup scripts
├── ⚙️ backend/              # FastAPI server
├── 🎨 frontend/             # React web interface
├── 📋 config/               # Configuration files
└── 📚 docs/                 # Documentation
```

## 🏆 Target Performance

| Benchmark | Current Target | Stretch Goal |
|-----------|----------------|--------------|
| MMLU | >70% | 75%+ |
| HellaSwag | >85% | 90%+ |
| ARC-Challenge | >65% | 70%+ |
| TruthfulQA | >55% | 60%+ |
| GSM8K | >60% | 65%+ |
| HumanEval | >50% | 60%+ |

## 🔧 Troubleshooting

### Common Issues

1. **Dataset not found**
   ```bash
   python scripts/simple_download.py
   ```

2. **Dependencies missing**
   ```bash
   pip install -r requirements.txt
   ```

3. **CUDA out of memory**
   - Reduce batch size in config
   - Use gradient checkpointing
   - Try QLoRA fine-tuning

4. **Frontend not loading**
   ```bash
   cd frontend
   npm install --legacy-peer-deps
   ```

## 🚀 Development Workflow

### 1. Data Preparation
```bash
python scripts/simple_download.py
```

### 2. Model Training
```bash
python scripts/train_model.py --config config/alpaca_7b.yaml
```

### 3. Evaluation
```bash
python scripts/evaluate_model.py --model-path experiments/alpaca-domination-7b
```

### 4. Inference Testing
```bash
python backend/main.py
```

## 🎯 Key Features Implemented

✅ **Core Architecture**
- Advanced transformer with Grouped Query Attention
- Flash Attention integration
- RoPE positional embeddings
- Constitutional AI components

✅ **Training Pipeline**
- Mixed precision training
- Gradient checkpointing
- DeepSpeed optimization
- LoRA/QLoRA support

✅ **Web Interface**
- Real-time streaming chat
- Dark/light mode
- Model selection
- Performance monitoring

✅ **API Server**
- OpenAI-compatible endpoints
- WebSocket support
- Performance metrics
- Model management

## 🎪 Demo Usage

### API Testing

```bash
curl -X POST "http://localhost:8000/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "messages": [
         {"role": "user", "content": "What makes you the best Alpaca model?"}
       ],
       "max_tokens": 512,
       "temperature": 0.7
     }'
```

### Python Client

```python
import requests

response = requests.post(
    "http://localhost:8000/chat/completions",
    json={
        "messages": [
            {"role": "user", "content": "Explain quantum computing"}
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

## 🏗️ Next Steps

1. **Complete Training**: Run full training on the Alpaca dataset
2. **Benchmark Testing**: Evaluate against standard benchmarks
3. **Optimization**: Implement quantization and optimization
4. **Deployment**: Deploy to cloud platforms
5. **Community**: Share results with the AI community

## 🎯 Success Metrics

- [x] Project structure created
- [x] Dataset downloaded and processed
- [x] Model architecture implemented
- [x] Training pipeline ready
- [x] API server functional
- [x] Web interface created
- [ ] Model training completed
- [ ] Benchmark evaluation
- [ ] Performance optimization
- [ ] Production deployment

## 🆘 Getting Help

- **Documentation**: Check the `docs/` directory
- **Issues**: Review error messages and logs
- **Community**: Join AI/ML communities for support
- **Resources**: Refer to HuggingFace documentation

## 🌟 Contributing

This project aims to push the boundaries of what's possible with Alpaca-trained models. Contributions welcome!

---

**Ready to dominate? Let's build the world's best Alpaca model! 🏆**
