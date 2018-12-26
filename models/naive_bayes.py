import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
from sklearn.naive_bayes import BernoulliNB
import psycopg2

import sys

class NaiveBayesClassifier:
    def __init__(self, db_credentials = {'user' : os.environ.get('DB_USER'), 'password' : os.environ.get('DB_PASSWORD'),
            'host' : os.environ.get('DB_HOST'), 'port' : os.environ.get('DB_PORT'), 'database' : os.environ.get('DB_DATABASE')},
            data_source = 'csv', tabu_file_path = '/data/naive_bayes/interim/tabu.txt', csv_file_path= '/data/naive_bayes/raw/spam.csv'):
        self.NaiveBayes = None
        self.tabu = dict()
        self.tabu_length = len(self.tabu)
        self.tabu_list = []
        self.db_credentials = db_credentials
        self.connection = psycopg2.connect(user = db_credentials['user'],
            password = db_credentials['password'],
            host = db_credentials['host'],
            port = db_credentials['port'],
            database = db_credentials['database'])
        self.cursor = self.connection.cursor()
        self.data_source = data_source
        self.PROJ_ROOT = os.path.abspath(os.path.join(os.pardir, os.pardir))
        self.csv_file_path = self.PROJ_ROOT + csv_file_path # path = file name to use for exporting list

    def read_csv_data(self):
        df = pd.read_csv(self.csv_file_path, usecols=[0,1], encoding='latin-1')
        df.columns=['label','content']
        n = int(df.shape[0])
        # split into training data and test data
        return self.split_train_test(df)
    
    def read_sql_data(self):
        import pandas.io.sql as psql
        # SMS_df = pd.read_sql(PROJ_ROOT +'/data/naive_bayes/raw/spam.csv',usecols=[0,1],encoding='latin-1')
        df = psql.read_sql("SELECT suspicious, message FROM giftcertificates", self.connection)
        df.columns=['label','content']
        n = int(df.shape[0])
        # split into training data and test data
        return self.split_train_test(df)


    def split_train_test(self, df, train_size=0.8):
        """ Splits data into train and test dataframes. Defaults to 80 20 split if not specified"""
        split_df = pd.DataFrame(np.random.randn(df.shape[0], 2))
        msk = np.random.rand(len(df)) < train_size
        train = df[msk]
        test = df[~msk]
        return train, test

    def close_connection(self):
        if(self.connection):
            self.cursor.close()
            self.connection.close()
            self.log.info("PostgreSQL connection is closed")

    def generate_tabu_list(self, tabu_size=300,ignore=3, spam_label='spam', valid_label='ham'):
        """
        tabu_size = length of exported list (ie how many rows through dataframe that function will process)
        ignore = minimum word length necessary to process word (ie ignore common short words like a, at, the, etc)
        """
        if self.data_source is 'csv': train_df,_ = self.read_csv_data()
        if self.data_source is 'sql': train_df,_ = self.read_sql_data()
        # else:
        #      train_df,_ = self.read_sql_data()
        spam_TF_dict = dict()
        valid_TF_dict = dict()
        IDF_dict = dict()

        # ignore all other than letters.
        # returns list of words downcased, removing punctuation and anything that is not a letter
        train_df['cleaned_content'] = train_df.content.apply( lambda x: [i.lower() for i in re.findall('[A-Za-z]+', re.sub("'","",x))])
        
    #   # go through each word in the dataset and add it to a dict of 
        for i in range(train_df.shape[0]):
            if train_df.iloc[i].label == spam_label:
                for find in train_df.iloc[i].cleaned_content:
                    if len(find)<ignore: continue
                    try:
                        # if the current word is already in our spam dict, increment the value (ie number of
                        # occurences) by 1
                        spam_TF_dict[find] = spam_TF_dict[find] + 1
                    except:	
                        # if the current word is not in our spam dict, add it, set the initial value to 1
                        # and add the word to our valid dict and set the value to 0
                        spam_TF_dict[find] = spam_TF_dict.get(find,1)
                        valid_TF_dict[find] = valid_TF_dict.get(find,0)
            # valid label
            else:
                for find in train_df.iloc[i].cleaned_content:
                    if len(find)<ignore: continue
                    try:
                        valid_TF_dict[find] = valid_TF_dict[find] + 1
                    except:	
                        spam_TF_dict[find] = spam_TF_dict.get(find,0)
                        valid_TF_dict[find] = valid_TF_dict.get(find,1)

            # basically just a list of each unique word
            word_set = set()
            for find in train_df.iloc[i].cleaned_content:
                if len(find)<ignore: continue
                if not(find in word_set):
                    try:
                        IDF_dict[find] = IDF_dict[find] + 1
                    except:	
                        IDF_dict[find] = IDF_dict.get(find,1)
                word_set.add(find)

        word_df = pd.DataFrame(list(zip(valid_TF_dict.keys(),valid_TF_dict.values(),spam_TF_dict.values(),IDF_dict.values())))
        word_df.columns = ['keyword','valid_TF','spam_TF','IDF']
        word_df['valid_TF'] = word_df['valid_TF'].astype('float')/train_df[train_df['label']==valid_label].shape[0]
        word_df['spam_TF'] = word_df['spam_TF'].astype('float')/train_df[train_df['label']==spam_label].shape[0]
        word_df['IDF'] = np.log10(train_df.shape[0]/word_df['IDF'].astype('float'))
        word_df['valid_TFIDF'] = word_df['valid_TF']*word_df['IDF']
        word_df['spam_TFIDF'] = word_df['spam_TF']*word_df['IDF']
        word_df['diff']=word_df['spam_TFIDF']-word_df['valid_TFIDF']

        selected_spam_key = word_df.sort_values('diff',ascending=False)

        print('>>>Generating Tabu List...\n  Tabu List Size: {}\n   The words shorter than {} are ignored by model\n'.format(tabu_size, ignore))
        tabu_list = []
        for word in selected_spam_key.head(tabu_size).keyword:
            tabu_list.append(word)
        return tabu_list


    def read_tabu_list(self, tabu_list):
        keyword_dict = dict()
        i = 0
        for word in tabu_list:
            keyword_dict.update({word.strip():i})
            i+=1
        self.tabu = keyword_dict
        self.tabu_length = len(self.tabu)
        return keyword_dict, len(keyword_dict)

    # create a numpy array of length tabu, ie the number of unique words
    # go through each word passed in 'content' (content is a string of words)
    # for each unique word in the string, find it's index in the numpy array,
    # and set its value to '1' to show it exists
    def convert_content(self, content, tabu):
        self.tabu_length = len(tabu)
        res = np.int_(np.zeros(self.tabu_length))
        finds = re.findall('[A-Za-z]+', content)
        for find in finds:
            find=find.lower()
            try:
                i = tabu[find]
                res[i]=1
            except:
                continue
        return res

    def learn(self, spam_label='spam'):
        tabu = self.tabu
        if self.data_source is 'csv': train,_ = self.read_csv_data()
        if self.data_source is 'sql': train,_ = self.read_sql_data()
        n = train.shape[0]
        X = np.zeros((n,self.tabu_length)); Y=np.int_(train.label==spam_label)
        for i in range(n):
            X[i,:] = self.convert_content(train.iloc[i].content, tabu)

        NaiveBayes = BernoulliNB()
        NaiveBayes.fit(X, Y)

        Y_hat = NaiveBayes.predict(X)
        print('>>>Learning...\n  Learning Sample Size: {}\n  Accuarcy (Training sample): {:.2f}％\n'.format(n,sum(np.int_(Y_hat==Y))*100./n))
        return NaiveBayes

    def test(self, NaiveBayes, spam_label='spam'):
        if self.data_source is 'csv': train,_ = _,test = self.read_csv_data()
        if self.data_source is 'sql': train,_ = _,test = self.read_sql_data()
        n = test.shape[0]
        X = np.zeros((n,self.tabu_length)); Y=np.int_(test.label==spam_label)
        for i in range(n):
            X[i,:] = self.convert_content(test.iloc[i].content, self.tabu)
        Y_hat = NaiveBayes.predict(X)
        print ('>>>Cross Validation...\n  Testing Sample Size: {}\n  Accuarcy (Testing sample): {:.2f}％\n'.format(n,sum(np.int_(Y_hat==Y))*100./n))
        return

    def predictSMS(self, SMS):
        X = self.convert_content(SMS, self.tabu)
        Y_hat = self.NaiveBayes.predict(X.reshape(1,-1))
        if int(Y_hat) == 1:
            print ('SPAM: {}'.format(SMS))
        else:
            print ('HAM: {}'.format(SMS))

    def predict_fraud(self, message):
        X = self.convert_content(message, self.tabu)
        Y_hat = self.NaiveBayes.predict(X.reshape(1,-1))
        if int(Y_hat) == 1:
            print ('Fraud: {}'.format(message))
        else:
            print ('Not Fraud: {}'.format(message))