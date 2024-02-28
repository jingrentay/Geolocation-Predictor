from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from torchvision import transforms
from model import *

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}


# Load your pre-trained model
checkpoint = torch.load("image_gps_modellite.pth", map_location=('cpu'))
model = ImageGPSModelV3()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        prediction = model(image)
    return prediction.squeeze().tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = app.config['UPLOAD_FOLDER'] + '/' + filename
        file.save(filepath)

        prediction = predict_image(filepath)

        return render_template('index.html', filename=filename, prediction=prediction)

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
