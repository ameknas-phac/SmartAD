import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import pandas as pd
# from model import model_predict  # Import your model prediction function

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
            query_sequences = query_df['sequence'].tolist()

            # Read the reference sequence and strain name
            reference_df = pd.read_csv(reference_filepath)
            if 'strain' not in reference_df.columns or 'sequence' not in reference_df.columns:
                flash('Reference CSV must have "strain" and "sequence" columns.')
                return redirect(url_for('home'))
            reference_strain = reference_df['strain'].tolist()[0]
            reference_sequence = reference_df['sequence'].tolist()[0]

            # Perform predictions
            predictions = []
            for strain, seq in zip(query_strains, query_sequences):
                # Replace with your actual prediction logic
                # prediction = model_predict(seq, reference_sequence)
                prediction = compute_antigenic_distance(seq, reference_sequence)
                predictions.append({'strain': strain, 'prediction': prediction})

            # Pass results to the result template
            return render_template('result.html', predictions=predictions, reference_strain=reference_strain)
        except Exception as e:
            flash(f"An error occurred while processing the files: {e}")
            return redirect(url_for('home'))

    return render_template('index.html')

# Placeholder function for antigenic distance calculation
def compute_antigenic_distance(seq1, seq2):
    # Check if sequences are of the same length, otherwise skip prediction
    if len(seq1) != len(seq2):
        return None  # Return None for sequences of different lengths
    distance = hamming_distance(seq1, seq2)
    return distance

def hamming_distance(seq1, seq2):
    # Ensure sequences are of equal length
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)