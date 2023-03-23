from flask import Flask,render_template,request
import pickle
import numpy as np
app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('BMI.html')

@app.route('/predict',methods=["POST"])
def predict():
    feature=[int(x) for x in request.form.values()]
    feature_final=[np.array(feature)]
    prediction=model.predict(feature_final)
    output=round(prediction[0],1)
    return render_template('BMI.html',prediction_text='BMI of the person is. {}'.format(output))
if __name__=="__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
