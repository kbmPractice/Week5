import numpy as np
from flask import Flask, request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('iris_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('iris_index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    #output = round(prediction[0], 2)
    iris_names= ['Setosa','Versicolor','Virginica']
    #wine_class = ['class_0', 'class_1', 'class_2']
    output = iris_names[int(prediction[0])]
    #output = wine_class[int(prediction[0])]

    return render_template('iris_index.html', prediction_text='Predicted Iris Class {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)