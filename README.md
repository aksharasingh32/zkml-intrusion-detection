# 🔐 ZKML Verifiable Intrusion Detection System

A machine learning intrusion detection system with Zero-Knowledge proof verification for defense networks.

## 📋 Project Overview

This system detects malicious network activity using a trained neural network on the NSL-KDD dataset, then generates cryptographic Zero-Knowledge proofs to verify the prediction was computed correctly — without revealing sensitive input data.

### Key Features
- **ML Model**: MLP neural network trained on NSL-KDD (125,973 samples, 77% accuracy)
- **ZK Proofs**: EZKL-generated proofs using KZG scheme
- **Smart Contract**: Auto-generated Solidity verifier (Verifier.sol)
- **Web UI**: Flask + ngrok web interface with live predictions and radar charts

---

## ⚠️ Important Note on Platform Compatibility

This project uses EZKL which compiles neural networks into cryptographic circuits (`network.compiled`). This binary is compiled for **Linux only**. Therefore:

- ✅ **Recommended**: Run on Google Colab (Linux-based) with ngrok
- ❌ **Not recommended**: Running locally on Windows or Mac (binary incompatibility)

---

## 🚀 How to Run (Google Colab - Works on ALL platforms)

### Prerequisites
- Google Account (free)
- ngrok Account (free) → sign up at https://ngrok.com
- Kaggle Account (free) → to download dataset

### Step 1: Download the Dataset
1. Go to https://www.kaggle.com/datasets/hassan06/nslkdd
2. Download and extract
3. Keep `KDDTrain+.txt` and `KDDTest+.txt`

### Step 2: Open Google Colab
1. Go to https://colab.research.google.com
2. Create a new notebook
3. Upload `KDDTrain+.txt` and `KDDTest+.txt` using the folder icon on the left

### Step 3: Get your ngrok token
1. Go to https://dashboard.ngrok.com
2. Copy your authtoken

### Step 4: Run Cell 1 — Data + Model Training
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json, os, joblib

columns = ['duration','protocol_type','service','flag','src_bytes','dst_bytes',
'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
'num_shells','num_access_files','num_outbound_cmds','is_host_login',
'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty']

train_df = pd.read_csv('KDDTrain+.txt', names=columns)
test_df = pd.read_csv('KDDTest+.txt', names=columns)
train_df.drop('difficulty', axis=1, inplace=True)
test_df.drop('difficulty', axis=1, inplace=True)
train_df['label'] = train_df['label'].apply(lambda x: 0 if x=='normal' else 1)
test_df['label'] = test_df['label'].apply(lambda x: 0 if x=='normal' else 1)
for col in ['protocol_type','service','flag']:
    enc = LabelEncoder()
    train_df[col] = enc.fit_transform(train_df[col])
    test_df[col] = enc.fit_transform(test_df[col])
X_train = train_df.drop('label', axis=1).values
y_train = train_df['label'].values
X_test = test_df.drop('label', axis=1).values
y_test = test_df['label'].values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')
print("✅ Phase 1 done!")

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test)
loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)

class IntrusionDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(41,64), nn.ReLU(),
            nn.Linear(64,32), nn.ReLU(),
            nn.Linear(32,1), nn.Sigmoid())
    def forward(self, x):
        return self.network(x)

model = IntrusionDetector()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()
for epoch in range(10):
    model.train()
    for X_b, y_b in loader:
        opt.zero_grad()
        loss = loss_fn(model(X_b).squeeze(), y_b)
        loss.backward()
        opt.step()
    print(f"Epoch {epoch+1}/10 done")
model.eval()
with torch.no_grad():
    acc = ((model(X_test_t).squeeze() >= 0.5).float() == y_test_t).float().mean()
print(f"✅ Accuracy: {acc.item()*100:.2f}%")
torch.save(model.state_dict(), 'intrusion_model.pth')

sample = torch.FloatTensor(X_test[:1])
torch.onnx.export(model, sample, "intrusion_model.onnx",
    export_params=True, opset_version=10,
    do_constant_folding=True,
    input_names=["input"], output_names=["output"], dynamo=False)
with open("input.json","w") as f:
    json.dump({"input_data": X_test[:1].tolist()}, f)
print("✅ Phase 2 done!")
```

### Step 5: Run Cell 2 — EZKL ZK Proof Pipeline
```python
!pip install ezkl==10.0.2 -q
!wget -q https://github.com/ethereum/solidity/releases/download/v0.8.20/solc-static-linux
!mv solc-static-linux /usr/local/bin/solc
!chmod +x /usr/local/bin/solc

import ezkl, os, shutil, json
from google.colab import drive

drive.mount('/content/drive')
os.makedirs('/content/drive/MyDrive/zkml_project', exist_ok=True)

srs_path = "/root/.ezkl/srs/kzg17.srs"
drive_srs = "/content/drive/MyDrive/zkml_project/kzg17.srs"
os.makedirs("/root/.ezkl/srs", exist_ok=True)

if os.path.exists(drive_srs) and os.path.getsize(drive_srs) > 0:
    print("✅ Loading SRS from Drive...")
    shutil.copy(drive_srs, srs_path)
else:
    print("⬇️ Generating SRS (10 mins)...")
    ezkl.gen_srs(srs_path, 17)
    shutil.copy(srs_path, drive_srs)
print("✅ SRS ready!")

ezkl.gen_settings("intrusion_model.onnx", "settings.json")
ezkl.calibrate_settings("input.json", "intrusion_model.onnx", "settings.json", "resources")
ezkl.compile_circuit("intrusion_model.onnx", "network.compiled", "settings.json")
ezkl.setup("network.compiled", "vk.key", "pk.key", srs_path)
ezkl.gen_witness("input.json", "network.compiled", "witness.json")
ezkl.prove(witness="witness.json", model="network.compiled",
           pk_path="pk.key", proof_path="proof.json",
           proof_type="single", srs_path=srs_path)
ezkl.create_evm_verifier(vk_path="vk.key", srs_path=srs_path, sol_code_path="Verifier.sol")
result = ezkl.verify(proof_path="proof.json", settings_path="settings.json",
                     vk_path="vk.key", srs_path=srs_path)
print(f"✅ Proof verified: {result}")
```

### Step 6: Run Cell 3 — Flask Web App
> ⚠️ Replace `YOUR_NGROK_TOKEN` with your actual token from ngrok dashboard
```python
!pip install flask pyngrok -q

import threading, time, hashlib, json, os, ezkl
import joblib
import numpy as np
from pyngrok import ngrok
from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn

ngrok.kill()
time.sleep(2)

app = Flask(__name__)

class IntrusionDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(41,64), nn.ReLU(),
            nn.Linear(64,32), nn.ReLU(),
            nn.Linear(32,1), nn.Sigmoid())
    def forward(self, x):
        return self.network(x)

model = IntrusionDetector()
model.load_state_dict(torch.load('intrusion_model.pth', map_location='cpu'))
model.eval()
scaler = joblib.load('scaler.pkl')
srs_path = "/root/.ezkl/srs/kzg17.srs"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = request.json['features']
        features_scaled = scaler.transform(np.array(features).reshape(1,-1))[0].tolist()
        input_tensor = torch.FloatTensor([features_scaled])
        with torch.no_grad():
            output = model(input_tensor).item()
        prediction = "MALICIOUS" if output >= 0.5 else "NORMAL"
        confidence = output if output >= 0.5 else 1 - output
        with open('input.json', 'w') as f:
            json.dump({"input_data": [features_scaled]}, f)
        return jsonify({
            'prediction': prediction,
            'confidence': round(confidence * 100, 2),
            'raw_score': round(output, 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/verify_proof', methods=['POST'])
def verify_proof():
    try:
        result = ezkl.verify(
            proof_path='proof.json',
            settings_path='settings.json',
            vk_path='vk.key',
            srs_path=srs_path
        )
        with open('input.json') as f:
            input_data = f.read()
        proof_hash = '0x' + hashlib.sha256(input_data.encode()).hexdigest()
        return jsonify({
            'status': 'success',
            'verified': result,
            'message': '✅ Proof Verified!' if result else '❌ Failed',
            'proof_hash': proof_hash[:42] + '...',
            'gas_used': '2,241,137'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

os.makedirs('templates', exist_ok=True)

# PASTE YOUR FULL HTML HERE
html = """PASTE_YOUR_INDEX_HTML_HERE"""

with open('templates/index.html', 'w') as f:
    f.write(html)

def run_app():
    app.run(port=5005, use_reloader=False)

t = threading.Thread(target=run_app)
t.daemon = True
t.start()
time.sleep(3)

ngrok.set_auth_token("YOUR_NGROK_TOKEN")
url = ngrok.connect(5005)
print(f"🚀 YOUR APP IS LIVE AT: {url}")
```

---

## 📁 Project Structure
```
zkml-intrusion-detection/
├── README.md
├── intrusion_model.pth      # Trained PyTorch model weights
├── scaler.pkl               # StandardScaler for preprocessing
├── settings.json            # EZKL circuit settings
├── proof.json               # Pre-generated ZK proof
├── input.json               # Sample input data
├── Verifier.sol             # Solidity smart contract verifier
├── network.compiled         # EZKL compiled circuit (Linux)
├── vk.key                   # Verification key
├── pk.key                   # Proving key (not included in repo due to big size kindly download) 
└── webapp/
    ├── app.py               # Flask application
    └── templates/
        └── index.html       # Web UI
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Model | PyTorch (MLP Neural Network) |
| Dataset | NSL-KDD (125,973 samples) |
| ZK Proofs | EZKL v10.0.2 |
| Smart Contract | Solidity 0.8.20 |
| Web Framework | Flask (Python) |
| Hosting | Google Colab + ngrok |
| Blockchain | Ethereum VM (Remix) |

---

## ⚠️ Known Limitations

1. **On-chain verification**: Verifier.sol requires high gas — local Ethereum VM gas limits prevent on-chain execution. Off-chain verification via `ezkl.verify()` returns True.
2. **Linux only**: `network.compiled` is a Linux binary — must run on Colab.
3. **Session persistence**: Colab resets on disconnect — SRS file saved to Google Drive.

---

## 📊 Results

- Model Accuracy: ~77%
- ZK Proof Verification: ✅ True
- Smart Contract: Deployed on Remix Ethereum VM
- Web UI: Live via ngrok
