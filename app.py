''' 
Decription: 
Arguments:
Output:
https://devhub.in.th/blog/flask-python
https://www.borntodev.com/2023/03/31/machine-learning-apis-%E0%B8%94%E0%B9%89%E0%B8%A7%E0%B8%A2-flask/
https://support.hostneverdie.com/index.php/knowledgebase/161/HTTP-Standard-or-GET-or-POST-or-PUT-or-PATCH-or-DELETE.html
To run app: flask --app app run

''' 
import time
from flask import Flask, render_template, request,flash, redirect, url_for,send_from_directory
# from flask_uploads import configure_uploads, UploadSet, DATA
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
import io
from forms import MyForm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,roc_curve
from plotly.io import to_html
import plotly.graph_objs as go
# from summarytools import dfSummary
from joblib import dump
import os
from newSVM import NewSVM, accuracy_plot,data_split_plot,cm_plot, hinge_plot, roc_plot, runtimes_plot


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to store uploaded files
app.config['UPLOAD_FOLDER_file'] = 'uploaded_file.csv'  # Folder to store uploaded files
app.config['MODEL_FOLDER'] = 'models'  # Folder to store model files
df=[]

# # Flask-Uploads configuration
# csvs = UploadSet('data', DATA)
# app.config['UPLOADED_DATA_DEST'] = 'uploads'
# configure_uploads(app, csvs)


@app.route('/form', methods=['GET', 'POST'])
def form():
    form = MyForm()
    if form.validate_on_submit():
        flash('Form submitted successfully')
        # Process the form data here if necessary
        return redirect(url_for('form'))
    return render_template('form.html', form=form)
@app.route('/')
def home():    
    return render_template("index2.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload-csv', methods=['POST'])
def upload_csv():
    global df
    # if 'file' not in request.files:
    #     return redirect(url_for('index'))
    if 'file' not in request.files:
        return render_template('error.html', error='No file part')
    

    file = request.files['file']
    header = request.form.get('header') == 'on'  # Check if the checkbox was ticked

    if file and file.filename.endswith('.csv'):
        try:
            # filepath = os.path.join(app.config['UPLOAD_FOLDER'], app.config['UPLOAD_FOLDER_file'])
            # file.save(filepath)
            if header:
                df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")), sep=",")
            else:
                df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")), sep=",", header=None)

            # df = pd.read_csv(filepath)
            profile = ProfileReport(df, explorative=True)
            profile.config.html.navbar_show=False
            profile_html = profile.to_html()
            return render_template('eda.html', eda_report=profile_html)
        except Exception as e:
            return render_template('error.html', error='Invalid CSV format and: '+str(e))

    else:
        return render_template('error.html', error='Unsupported file format')
    
@app.route('/training_results', methods=['POST'])
def next_page():
        # def train_and_get_results():
        global df

        # Split the data into train-test sets
        X = df.iloc[:, :-1].values.copy()
        y = df.iloc[:, -1].values.copy()
        split_proportion = float(request.form.get('split', 0.2)) 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_proportion, random_state=42)
        print('type(X_train): ',type(X_train))
        print('type(y_train): ',type(y_train))

        # Calculate train-test data information
        n_feature=X.shape[1]
        train_size = len(X_train)
        test_size = len(X_test)
        total_size = len(df)
        train_percent = (train_size / total_size) * 100
        test_percent = (test_size / total_size) * 100

        # Train the SVM model
        newSVM = NewSVM()
        newSVM.fit(X_train, y_train)

        model_directory = app.config['MODEL_FOLDER']
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        
        model_filename = 'new_svm_model.joblib'
        model_filepath = os.path.join(model_directory, model_filename)
        
        dump(newSVM, model_filepath)

        df_head_html = df.head().to_html(classes="table table-striped table-bordered")        
        data_split_plot_html = to_html(data_split_plot(train_size,test_size),full_html=False)

        y_pred = newSVM.predict(X_test)
        y_pred_c = np.where(y_pred >= 0.0, 1, -1)
        # y_pred_proba = newSVM.predict_proba(X_test)
        cm = confusion_matrix(y_test, y_pred_c.T)  

        cm_plot_html=to_html(cm_plot(cm),full_html=False) 

        # # Generate ROC Curve Plot
        # fpr, tpr, _ = roc_curve(y_test, y_pred.T)        
        # roc_plot_html = to_html(roc_plot(fpr,tpr), full_html=False)
        runtimes_plot_html = to_html(runtimes_plot(newSVM), full_html=False)        
        hinge_plot_html = to_html(hinge_plot(newSVM), full_html=False)
        accuracy_plot_html = to_html(accuracy_plot(newSVM), full_html=False)

        return render_template('results.html', 
                            confusion_matrix_plot=cm_plot_html,
                            runtimes_plot=runtimes_plot_html, 
                            training_hinge_plot=hinge_plot_html,
                            training_accuracy_plot=accuracy_plot_html,
                            data_split_plot=data_split_plot_html,
                            df_head=df_head_html,
                            # roc_curve_plot=roc_plot_html,
                            train_size=train_size, 
                            test_size=test_size,
                            total_size=train_size+test_size,
                            n_feature=n_feature,
                            train_percent=train_percent, 
                            test_percent=test_percent,
                            model_filename=model_filename)
        # threading.Thread(target=train_and_get_results).start()
        # return render_template('training_in_progress.html')
@app.route('/download-model/<path:filename>', methods=['GET'])
def download_model(filename):
    # Make sure the directory is correct and the file exists
    directory = app.config['MODEL_FOLDER']
    # return send_from_directory("..\\..\\"+directory, filename, as_attachment=True)
    # print("..\\"+directory)
    return send_from_directory(directory=directory, path=filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)