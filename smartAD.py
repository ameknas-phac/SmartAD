from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
from metafluad_TEST import metafluad_model 

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        query_file = request.files['query_csv']
        reference_file = request.files['reference_csv']
        
        # Validate file uploads
        if query_file.filename == '' or not allowed_file(query_file.filename):
            flash('Query file is required and must be a CSV.')
            return redirect(request.url)
        if reference_file.filename == '' or not allowed_file(reference_file.filename):
            flash('Reference file is required and must be a CSV.')
            return redirect(request.url)

        # Secure filenames and save the files
        query_filename = os.path.join(app.config['UPLOAD_FOLDER'], query_file.filename)
        reference_filename = os.path.join(app.config['UPLOAD_FOLDER'], reference_file.filename)
        query_file.save(query_filename)
        reference_file.save(reference_filename)

        # Retrieve selected virus model from form
        virus_type = request.form.get('virus_type')

        # Load the correct model based on the selection
        models = {
            'H1N1': 'model_H1N1.pth',
            'H3N2': 'model_H3N2.pth',
            'H5N1': 'model_H5N1.pth',
            'Influenza_B': 'model_B-vic.pth',
        }
        
        model_file = models.get(virus_type)
        
        if not model_file:
            flash('Invalid virus model selection.')
            return redirect(request.url)

        try:
            # Initialize and load the selected model
            model = metafluad_model(model_file)  # Pass the selected model file to the model class
            
            # Use the model to predict distances
            query_df = pd.read_csv(query_filename)
            reference_df = pd.read_csv(reference_filename)
            query_strains = query_df['strain'].tolist()
            reference_strain = reference_df['strain'].tolist()[0]

            distances = model.distances(query_filename, reference_filename)

            # Prepare predictions and render the result page
            predictions = [{'strain': strain, 'prediction': distance} for strain, distance in zip(query_strains, distances)]
            return render_template('result.html', predictions=predictions, reference_strain=reference_strain)
        
        except Exception as e:
            flash(f"An error occurred: {e}")
            return redirect(request.url)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)