import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import pandas as pd
from metafluad_TEST import metafluad_model

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flashing messages

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if the files are part of the request
        if 'query_csv' not in request.files or 'reference_csv' not in request.files:
            flash('Both query and reference CSV files are required.')
            return redirect(request.url)

        query_file = request.files['query_csv']
        reference_file = request.files['reference_csv']

        # Validate query file
        if query_file.filename == '':
            flash('No query file selected.')
            return redirect(request.url)
        if not allowed_file(query_file.filename):
            flash('Query file must be a CSV.')
            return redirect(request.url)

        # Validate reference file
        if reference_file.filename == '':
            flash('No reference file selected.')
            return redirect(request.url)
        if not allowed_file(reference_file.filename):
            flash('Reference file must be a CSV.')
            return redirect(request.url)

        # Secure filenames and save files
        query_filename = secure_filename(query_file.filename)
        reference_filename = secure_filename(reference_file.filename)
        query_filepath = os.path.join(app.config['UPLOAD_FOLDER'], query_filename)
        reference_filepath = os.path.join(app.config['UPLOAD_FOLDER'], reference_filename)
        query_file.save(query_filepath)
        reference_file.save(reference_filepath)

        try:
            # Read the query sequences
            query_df = pd.read_csv(query_filepath)
            if 'strain' not in query_df.columns or 'sequence' not in query_df.columns:
                flash('Query CSV must have "strain" and "sequence" columns.')
                return redirect(url_for('home'))
            query_strains = query_df['strain'].tolist()

            # Read the reference sequence and strain name
            reference_df = pd.read_csv(reference_filepath)
            if 'strain' not in reference_df.columns or 'sequence' not in reference_df.columns:
                flash('Reference CSV must have "strain" and "sequence" columns.')
                return redirect(url_for('home'))
            reference_strain = reference_df['strain'].tolist()[0]

            # Use Adam's MetaFluAD model to compute the distances
            model = metafluad_model()  # Instantiate the model
            distances = model.distances(query_filepath, reference_filepath)  # Get antigenic distances

            # Prepare predictions with strains and their distances
            predictions = []
            for strain, distance in zip(query_strains, distances):
                predictions.append({'strain': strain, 'prediction': distance})

            # Pass results to the result template
            return render_template('result.html', predictions=predictions, reference_strain=reference_strain)
        except Exception as e:
            flash(f"An error occurred while processing the files: {e}")
            return redirect(url_for('home'))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)