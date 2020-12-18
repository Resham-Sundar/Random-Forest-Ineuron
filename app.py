from flask import Flask, render_template, request
from flask_cors import cross_origin
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__, template_folder='./templates', static_folder='./static')

Pkl_Filename = "model_rf.pkl" 
with open(Pkl_Filename, 'rb') as file:  
    model = pickle.load(file)
@app.route('/')

def hello_world():
    return render_template('home.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    crim = float(request.form['crim'])
    zn = float(request.form['zn'])
    indus = float(request.form['indus'])
    chas = float(request.form['chas'])
    nox = float(request.form['nox'])
    rm = float(request.form['rm'])
    age = float(request.form['age'])
    dis = float(request.form['dis'])
    rad = float(request.form['rad'])
    tax = float(request.form['tax'])
    ptratio = float(request.form['ptratio'])
    b = float(request.form['b'])
    lstat = float(request.form['lstat'])
    features = [[crim,zn,indus,chas,nox,rm,age,dis,rad,tax,ptratio,b,lstat]]
    print(features)
    pred = model.predict(features)[0]
    return render_template('op.html', pred=pred)

if __name__ == '__main__':
    app.run(debug=True)
