# Importing essential libraries and modules

from flask import Flask, render_template, request, redirect
from markupsafe import Markup
import numpy as np
import pandas as pd
from pathlib import Path
try:
    from App.utils.disease import disease_dic
    from App.utils.fertilizer import fertilizer_dic
    from App.utils.model import ResNet9
    from App import config
except ModuleNotFoundError:  # pragma: no cover - fallback for script execution
    from utils.disease import disease_dic
    from utils.fertilizer import fertilizer_dic
    from utils.model import ResNet9
    import config
import requests
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from requests import RequestException
# # ============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


# Loading crop recommendation model

crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    try:
        response = requests.get(complete_url, timeout=10)
        response.raise_for_status()
    except RequestException:
        return None

    x = response.json()

    if x.get("cod") == "404":
        return None

    main_block = x.get("main")
    if not main_block:
        return None

    temperature = main_block.get("temp")
    humidity = main_block.get("humidity")

    if temperature is None or humidity is None:
        return None

    temperature_celsius = round((temperature - 273.15), 2)
    return temperature_celsius, humidity


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    try:
        image = Image.open(io.BytesIO(img))
    except (UnidentifiedImageError, OSError):
        return None

    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    with torch.no_grad():
        yb = model(img_u)
        # Pick index with highest probability
        _, preds = torch.max(yb, dim=1)

    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'AgroBot - Home'
    return render_template('index.html', title=title)

# render crop recommendation form page


@ app.route('/crop-recommend')
def crop_recommend():
    title = 'AgroBot - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'AgroBot - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

# render disease prediction input page




# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'AgroBot - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")

        weather = weather_fetch(city)
        if weather is not None:
            temperature, humidity = weather
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop-result.html', prediction=final_prediction, title=title)

        else:

            return render_template('try_again.html', title=title)

# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'AgroBot - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    fertilizer_csv = Path(__file__).resolve().parents[1] / 'Data' / 'fertilizer.csv'
    df = pd.read_csv(fertilizer_csv)

    crop_requirements = df[df['Crop'] == crop_name]
    if crop_requirements.empty:
        error_message = Markup("We could not find fertilizer data for the selected crop. Please try another crop.")
        return render_template('fertilizer-result.html', recommendation=error_message, title=title)

    nr = crop_requirements['N'].iloc[0]
    pr = crop_requirements['P'].iloc[0]
    kr = crop_requirements['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    differences = [(abs(n), "N"), (abs(p), "P"), (abs(k), "K")]
    max_value = max(differences, key=lambda item: item[0])[1]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)

# render disease prediction result page


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'AgroBot - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        img = file.read()

        prediction_key = predict_image(img)

        if not prediction_key:
            error_message = Markup("We were unable to process the uploaded image. Please try again with a clear plant image.")
            return render_template('disease.html', title=title, error=error_message)

        disease_description = disease_dic.get(prediction_key)
        if disease_description is None:
            error_message = Markup("We could not find details for the predicted disease. Please try again later.")
            return render_template('disease.html', title=title, error=error_message)

        prediction = Markup(str(disease_description))
        return render_template('disease-result.html', prediction=prediction, title=title)
    return render_template('disease.html', title=title)


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)
