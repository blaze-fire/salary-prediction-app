import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

class Preprocess:

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub('[^a-zA-Z]', ' ', text)
        
        stopwords_dict = {word: 1 for word in stopwords.words("english")}
        text = " ".join([word for word in text.split() if word not in stopwords_dict])
           
        text = ' '.join([lemmatizer.lemmatize(w) for w in text.split()])

        return text
    
    def clean_data(self, df):
       
        df['Job_position'] = df['Job_position'].apply(lambda x: str(x).replace('\nnew','').lower())

        df.replace('na', np.nan, inplace=True)
        df['rating'] = df['rating'].astype('float')


        return df


    def work_location(self, df):

        work_type = []
        for j in range(len(df)):

            if re.findall('full.time?', df['requirements'][j].lower()):
                work_type.append(2)

            elif re.findall('remote?', df['requirements'][j].lower()):
                work_type.append(1)

            else:
                work_type.append(0)


        df['work_category'] = work_type

        return df


    #Educational criteria mentioned by these companies can also be useful
    def education(self, df):

        education_list = []
        education_dict = {'bachelor':1, 'master':2, 'graduate':3}
        for j in range(len(df)):
            
            if re.findall(r'(graduate|bachelor|master)', str(df['experience'][j]).lower()):
                education_list.append( education_dict[re.search(r'(graduate|bachelor|master)', df['experience'][j].lower()).group()] )
            
            elif re.findall(r'(graduate|bachelor|master)', df['requirements'][j].lower()):
                education_list.append( education_dict[re.search(r'(graduate|bachelor|master)', df['requirements'][j].lower()).group()] )
            
            else:
                education_list.append(0)


        df['education_level'] = education_list

        return df



    def seniority(self, df):

        seniority_list=[]
        for j in range(len(df)):

            if re.findall(r'senior', df['requirements'][j].lower()):
                seniority_list.append(2)
            
            elif re.findall(r'junior', df['requirements'][j].lower()):
                seniority_list.append(1)

            else:
                seniority_list.append(0)

        df['job_title'] = seniority_list

        return df


    def get_states(self, df):

        with open('utils/states.txt', 'r') as f:
            states = f.read()
            states_list = states.split(',')
        f.close()
        
        job_states = []

        for j in range(len(df)):
            counter = 0
            
            for i in states_list:
                if re.findall(i, str(df['Location'][j]).lower()):
                    job_states.append(states_list.index(i))
                    counter = 1
                    break
            
            if counter == 0:
                job_states.append(states_list.index('State_missing'))

        df['State'] = job_states
        
        return df




    def city(self, df):

        with open('utils/cities.txt', 'r') as f:
            cities = f.read()
            cities_list = cities.split(',')
        f.close()

        job_cities = []

        for j in range(len(df)):
            counter = 0
            
            for i in cities_list:
                if re.findall(i, str(df['Location'][j]).lower()):
                    job_cities.append(cities_list.index(i))
                    counter = 1
                    break
            
            if counter == 0:
                job_cities.append(cities_list.index('city_missing'))
                
        df['Cities'] = job_cities
        
        return df
    

    def final_operations(self, df):
    
        # remove columns with constant values
        df['Company'] = df['Company'].apply(lambda x: self.preprocess_text(str(x)))
        df['Job_position'] = df['Job_position'].apply(lambda x: self.preprocess_text(str(x)))
        df['requirements'] = df['requirements'].fillna('')
        df['requirements'] = df['requirements'].apply(lambda x: self.preprocess_text(str(x)))
        
        df['experience'] = df['experience'].fillna('')
        df['experience'] = df['experience'].apply(lambda x: self.preprocess_text(str(x)))
        df.drop(['Location'], axis=1, inplace=True)


        return df
    
    def __call__(self, df):
        df = self.clean_data(df)
        df = self.work_location(df)
        df = self.education(df)
        df = self.seniority(df)
        df = self.get_states(df)
        df = self.city(df)
        df = self.final_operations(df)
        return df