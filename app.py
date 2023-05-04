from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('nbamodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    #values are taken from the user inputs
    fgm_home = int(request.form["fgm_home"])
    fga_home = int(request.form["fga_home"])
    ftm_home = int(request.form["ftm_home"])
    fta_home = int(request.form["fta_home"])
    oreb_home = int(request.form["oreb_home"])
    dreb_home = int(request.form["dreb_home"])
    ast_home = int(request.form["ast_home"])
    stl_home = int(request.form["stl_home"])
    blk_home = int(request.form["blk_home"])
    tov_home = int(request.form["tov_home"])


    prediction = model.predict([[ fgm_home, fga_home,ftm_home,fta_home,oreb_home,dreb_home,ast_home,stl_home,blk_home,tov_home]])  # this returns a list e.g. [127.20488798], so pick first element [0]
    output = round(prediction[0], 2) 

    return render_template('index.html', prediction_text=f'The model predicts using the values you inputted your team will score {output} points!')

if __name__ == "__main__":
    app.run()
