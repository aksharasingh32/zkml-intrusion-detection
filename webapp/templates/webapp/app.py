from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import json
import os
import ezkl
import joblib

app = Flask(__name__)

# ---------------- Model ----------------
class IntrusionDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(41, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# Load model
model = IntrusionDetector()
model.load_state_dict(torch.load('../intrusion_model.pth', map_location='cpu'))
model.eval()

# Load scaler (VERY IMPORTANT)
scaler = joblib.load('../scaler.pkl')

# ---------------- Routes ----------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # 🔥 Apply SAME preprocessing as training
        features = np.array(data['features']).reshape(1, -1)
        features = scaler.transform(features)
        features = features.tolist()[0]

        # Model prediction
        input_tensor = torch.FloatTensor([features])
        with torch.no_grad():
            output = model(input_tensor).item()

        prediction = "MALICIOUS 🚨" if output >= 0.5 else "NORMAL ✅"
        confidence = output if output >= 0.5 else 1 - output

        # Save input for EZKL
        with open('../input.json', 'w') as f:
            json.dump({"input_data": [features]}, f)

        return jsonify({
            'prediction': prediction,
            'confidence': round(confidence * 100, 2),
            'raw_score': round(output, 4)
        })

    except Exception as e:
        print("🔥 ERROR in /predict:", e)
        return jsonify({'error': str(e)}), 500


@app.route('/verify_proof', methods=['POST'])
def verify_proof():
    try:
        srs_path = os.path.expanduser('~/.ezkl/srs/kzg17.srs')

        # Generate witness
        print("🟡 Generating witness...")
        ezkl.gen_witness('../input.json', '../network.compiled', '../witness.json')

        # Generate proof
        print("🟡 Generating proof...")
        ezkl.prove(
            witness='../witness.json',
            model='../network.compiled',
            pk_path='../pk.key',
            proof_path='../proof.json',
            proof_type='single',
            srs_path=srs_path
        )

        # Verify proof
        print("🟡 Verifying proof...")
        result = ezkl.verify(
            proof_path='../proof.json',
            settings_path='../settings.json',
            vk_path='../vk.key',
            srs_path=srs_path
        )

        return jsonify({
            'status': 'success',
            'verified': result,
            'message': '✅ Proof Verified!' if result else '❌ Verification Failed'
        })

    except Exception as e:
        print("🔥 ERROR in /verify_proof:", e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
