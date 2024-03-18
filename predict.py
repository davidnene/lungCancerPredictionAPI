import torch
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from PIL import Image

class_name = ['adenocarcinoma',
              'large.cell.carcinoma',
              'normal',
              'squamous.cell.carcinoma']

def predict(image, model):

    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    try:
      image = prediction_transform(image)[:3,:,:].unsqueeze(0)
    except:
      image = image.convert('RGB')
      image = prediction_transform(image)[:3,:,:].unsqueeze(0)


    with torch.no_grad():
      model.eval()
      pred = model(image)


    idx = torch.argmax(pred)

    prob = pred[0][idx].item()*100

    return prob, class_name[idx]

def ml_app(path, model):
    img = Image.open(path)
    
    prob, result = predict(img, model)
    return prob, result
