from flask import Flask, request, jsonify
import timm
import torch
import numpy as np
from predict import ml_app

app = Flask(__name__)


def predict_image(image):
    # Your model prediction logic here
    # For demonstration, let's assume it returns a dummy prediction
    return {"prediction": "dummy_prediction"}

model_test = timm.create_model('resnet50', pretrained=True)

model_test.fc = torch.nn.Sequential(torch.nn.Linear(2048, 256),
                                    torch.nn.Dropout(0.2),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(256, 64),
                                    torch.nn.Dropout(0.2),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(64, 32),
                                    torch.nn.Dropout(0.2),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(32, 4),
                                    torch.nn.Softmax()
                                    )

model_test.load_state_dict(torch.load('model\model_ResNet50_acc_max.pt', map_location=torch.device('cpu')))

@app.route('/predict', methods=['POST'])
def predict():
    # Check if image file is present in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    # Read the image file from the request
    image = request.files['image']

    # Pass the image to the model for prediction
    prob ,prediction= ml_app(image, model_test)
    # Return the prediction in JSON format
    results = {
        'probaility': round(prob, 2),
        'prediction': prediction
    }
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
