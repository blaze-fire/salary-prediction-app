from flask import Flask, jsonify, request, render_template
import numpy as np
import pandas as pd
from utils.data_preprocessing import Preprocess
from flask_sqlalchemy import SQLAlchemy
from pickle import load
from scipy.sparse import hstack
import os

# Import for Migrations
from flask_migrate import Migrate, migrate
 

app = Flask(__name__)

# adding configuration for using a sqlite database
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'

uri = os.getenv("DATABASE_URL")  # or other relevant config var
if uri and uri.startswith("postgres://"):
    uri = uri.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = uri


# Creating an SQLAlchemy instance
db = SQLAlchemy(app)

# Settings for migrations
migrate = Migrate(app, db)

# Models
class Profile(db.Model):
    # Id : Field which stores unique id for every row in

    id = db.Column(db.Integer, primary_key=True)
    Job_position = db.Column(db.String(100), unique=False, nullable=False)
    Company = db.Column(db.String(100), unique=False, nullable=False) 
    Location = db.Column(db.String(100), unique=False, nullable=False) 
    requirements = db.Column(db.String(3000), unique=False, nullable=False) 
    rating = db.Column(db.Float, unique=False)
    experience = db.Column(db.String(3000), unique=False, nullable=False) 
    
    # repr method represents how one object of this datatable
    # will look like
    def __repr__(self):
        return f"Company : {self.Company}, Job Profile: {self.Job_position}"


@app.route('/')
def home():
    return render_template('index.html')

# function to render index page
@app.route('/show')
def index():
    profiles = Profile.query.all()
    return render_template('show.html', profiles=profiles)


@app.route('/predict', methods=['POST'])
def predict():   
     
    x_in  = list(request.form.values())
    
    columns = ['Job_position', 'Company', 'Location', 'requirements', 'rating', 'experience']

    input_df = pd.DataFrame(columns = columns)

    for j in range(len(x_in)):
        input_df.loc[0, columns[j]] = x_in[j]    
            
    input_df['rating'] = pd.to_numeric(input_df['rating'])

    p = Profile(Job_position = input_df['Job_position'].values[0], Company=input_df['Company'].values[0], 
                Location=input_df['Location'].values[0], requirements = input_df['requirements'].values[0],
                rating = float(input_df['rating'].values[0]), experience = input_df['experience'].values[0], 
                )


    db.session.add(p)
    db.session.commit()


    # load the model
    model = load(open('models/model.pkl', 'rb'))
    
    # load the scaler
    transformer = load(open('models/scaler.pkl', 'rb'))

    input_df = Preprocess()(input_df)
    text_data = transformer.transform(input_df)
    num_cols = list(input_df.select_dtypes(exclude='object').columns)

    train_stack = hstack((input_df[num_cols].values, text_data))
    
    pred = model.predict(train_stack)
    prediction = np.round(pred, 2)

    return render_template('index.html', prediction_text='Your predicted annual salary is {}'.format(prediction[0]))



if __name__ == '__main__':
 app.run(debug=True)
