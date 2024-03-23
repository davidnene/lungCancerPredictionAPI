from flask import Flask, request, jsonify
import timm
import torch
import numpy as np
from predict import ml_app
from flask_cors import CORS
import base64

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins":"*"}})


# Create model blueprint
model = timm.create_model('resnet50', pretrained=True)

model.fc = torch.nn.Sequential(torch.nn.Linear(2048, 256),
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
# Load model
model.load_state_dict(torch.load('model\model_ResNet50_acc_max.pt', map_location=torch.device('cpu')))

@app.route('/predict', methods=['POST'])
def predict():
    # Check if image base64 is present in the request
    if 'cancer_image' not in request.get_json():
        return jsonify({'error': 'No image provided'})
    # Get the base64 string from the request
    photo = request.get_json()['cancer_image']
    
    # Decode the base64 string
    photo_data = base64.b64decode(photo)
    
    # create a png file and write bytes
    with open("image.png", "wb") as file:
        file.write(photo_data)

    img = 'image.png'

    # Pass the image to the model for prediction
    prob ,prediction= ml_app(img, model)
    
    # Return the prediction in JSON format
    results = {
        'probability': round(prob, 2),
        'prediction': prediction
    }
    return jsonify(results)


if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000)
