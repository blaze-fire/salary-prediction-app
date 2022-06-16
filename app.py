#Steps to push database schema chamges mentioned in Readme
 
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from utils.data_preprocessing import Preprocess
from flask_sqlalchemy import SQLAlchemy
from pickle import load
from scipy.sparse import hstack
import os

import io
import base64

#To download the data from database as csv
import flask_excel as excel

import re

#for plotting
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns


# Import for Migrations
from flask_migrate import Migrate, migrate


app = Flask(__name__)

# adding configuration for using a sqlite database
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'  #to run locally

uri = os.getenv("DATABASE_URL")  # or other relevant config var
if uri and uri.startswith("postgres://"):
    uri = uri.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = uri


# Creating an SQLAlchemy instance
db = SQLAlchemy(app)

# Settings for migrations
migrate = Migrate(app, db, render_as_batch=True)                                #https://stackoverflow.com/a/67277190/14204371

# Models
class Profile(db.Model):
    # Id : Field which stores unique id for every row in

    id = db.Column(db.Integer, primary_key=True)
    Job_position = db.Column(db.String(1000), unique=False, nullable=False)
    Company = db.Column(db.String(1000), unique=False, nullable=False)
    Location = db.Column(db.String(1000), unique=False, nullable=False)
    requirements = db.Column(db.String(10000), unique=False, nullable=False)
    rating = db.Column(db.Float, unique=False)
    experience = db.Column(db.String(10000), unique=False, nullable=False)

    # repr method represents how one object of this datatable
    # will look like
    def __repr__(self):
        return f"Company : {self.Company}, Job Profile: {self.Job_position}"


# function to render index page
@app.route('/')
def home():
    return render_template('index.html')


#function to download data as array
@app.route('/download', methods=['GET'])
def download_data():
    profiles = Profile.query.all()
    columns = ['Job_position', 'Company', 'Location',
               'requirements', 'rating', 'experience']


    df = []
    for data in profiles:
        df.append([data.Job_position, data.Company, data.Location, data.requirements, data.rating, data.experience])


    # To view data in a jupyter notebook
    # import pandas as pd
    # columns = ['Job_position', 'Company', 'Location', 'requirements', 'rating', 'experience']
    # x = pd.read_csv('export.csv', names=columns)

    excel.init_excel(app)
    extension_type = "csv"
    filename = "export" + "." + extension_type
    return excel.make_response_from_array(df, file_type=extension_type, file_name=filename)



def plotView(input_df):

    # predicted salary from '/predict'
    predictedSal = input_df['avg_yearly_sal'].values[0]

    df = pd.read_csv(r'data/data_for_eda.csv')
    df = df[df['avg_yearly_sal'] > 0]
    sal = df['avg_yearly_sal']
    sal.reset_index(drop=True, inplace=True)
    sal = sal.append(pd.Series({'avg_yearly_sal': predictedSal}), ignore_index=True)

    # Plot Salary Distribution
    fig = Figure(figsize=(10, 15))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_title("You vs Others")
    ax1.set_xlabel("Annual Average Salary (in crores)")
    ax1.set_ylabel("Count of Number of People")
    ax1.grid()

    # Generate bins with interval of 100000
    bins = np.arange(10000, 10000000, 100000)
    sns.histplot(ax=ax1, x=sal, bins=bins)

    # To get the first index of the bin where it is given than the current predicted salary
    idx = np.nonzero(bins > predictedSal)[0][0]
    ax1.patches[idx].set_facecolor('salmon')
    ax1.annotate(format(f'Rs. {predictedSal}'),
                     (ax1.patches[idx].get_x() + ax1.patches[idx].get_width() / 2., ax1.patches[idx].get_height()),
                     ha='center', va='center',
                     xytext=(0, 100),arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), textcoords='offset points')

    # ------------------------------------------------------------------------------------------------------------------------------

    # Skills vs salary plot (Bassically plot the average salary for the skills mentioned in the JD)
    text = input_df['requirements'].values[0]
    with open('utils/skills.txt') as f:
        skills = f.read()
    f.close()

    skills = skills.split(',')
    input_skills = list(set(re.findall(r"(?=(\b" + '\\b|\\b'.join(skills) + r"\b))", text)))

    df_skills = list(df['Job_position'].unique())
    input_skills = list(set(input_skills) & set(df_skills))

    # To build the dataframe of average salaries of skills mentioned in the requirements section
    plot_df = pd.DataFrame(columns=['skill', 'avg_yearly_sal'])
    for i in input_skills:
        avg_sal = df[df['Job_position'] == i]['avg_yearly_sal'].values.mean()
        plot_df = plot_df.append(
                                pd.Series({'skill': i, 'avg_yearly_sal': avg_sal}), 
                                ignore_index=True
                                )

    plot_df.sort_values(by='avg_yearly_sal', inplace=True)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_title("Average pay with these skills")
    ax2.grid()
    ax2 = sns.barplot(y='avg_yearly_sal', x='skill', data=plot_df, ax=ax2)
    for p in ax2.patches:
        ax2.annotate(format(p.get_height(), '.1f'),
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center',
                     xytext=(0, 9),
                     textcoords='offset points')
    ax2.set_ylabel("Annual Average Salary")

    # ------------------------------------------------------------------------------------------------------------------------------

    # Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)

    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

    return render_template("image.html", image=pngImageB64String, prediction=predictedSal)


@app.route('/predict', methods=['POST'])
def predict():

    x_in = list(request.form.values())

    columns = ['Job_position', 'Company', 'Location',
               'requirements', 'rating', 'experience']

    input_df = pd.DataFrame(columns=columns)

    for j in range(len(x_in)):
        input_df.loc[0, columns[j]] = x_in[j]

    input_df['rating'] = pd.to_numeric(input_df['rating'])

    p = Profile(Job_position=input_df['Job_position'].values[0],
                Company=input_df['Company'].values[0],
                Location=input_df['Location'].values[0],
                requirements=input_df['requirements'].values[0],
                rating=float(input_df['rating'].values[0]),
                experience=input_df['experience'].values[0],
                )

    db.session.add(p)
    db.session.commit()

    # load the model
    model = load(open('models/model.pkl', 'rb'))

    # load the scaler
    transformer = load(open('models/scaler.pkl', 'rb'))

    processed_df = Preprocess()(input_df)
    text_data = transformer.transform(processed_df)
    num_cols = list(processed_df.select_dtypes(exclude='object').columns)

    train_stack = hstack((processed_df[num_cols].values, text_data))

    pred = model.predict(train_stack)

    prediction = np.round(pred, 2)

    input_df['avg_yearly_sal'] = prediction[0]
    return plotView(input_df)


if __name__ == '__main__':
    app.run(debug=True)
