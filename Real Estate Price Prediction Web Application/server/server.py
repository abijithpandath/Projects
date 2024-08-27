import util
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.before_first_request
def initialize():
    util.load_saved_artifacts()

@app.route('/get_location_names')
def get_location_names():
    locations = util.get_location_names()
    if locations is None:
        return jsonify({'error': 'Failed to load locations'}), 500
    response = jsonify({
        'locations': locations
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    try:
        total_sqft = float(request.form['total_sqft'])
        location = request.form['location']
        bhk = int(request.form['bhk'])
        bath = int(request.form['bath'])

        estimated_price = util.get_estimated_price(location, total_sqft, bhk, bath)
        response = jsonify({
            'estimated_price': estimated_price
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction..")
    app.run()
