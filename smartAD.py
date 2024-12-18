import torch
import numpy as np
import random
import os
import gc
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from metafluad_TEST import metafluad_model
import csv

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def set_seed():
    """Set random seed for reproducibility on each request."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

@app.route('/', methods=['GET', 'POST'])
def home():
    # Set the seed at the beginning of each request
    set_seed()

    if request.method == 'POST':
        query_file = request.files['query_csv']
        reference_file = request.files['reference_csv']
        csv_only = request.form.get('csv_only')  # Get the CSV-only checkbox value
        
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

        # Retrieve selected virus model and batch size from the form
        virus_type = request.form.get('virus_type')
        batch_size = int(request.form.get('batch_size', 10))  # Default to 10 if not provided

        # Load the correct model based on the selection
        models = {
            'H1N1': 'model_H1N1.pth',
            'H3N2': 'model_H3N2.pth',
            'H5N1': 'model_H5N1.pth',
            'Influenza_B': 'model_B-vic.pth',
            'H3N2-IRVC': 'Smart_tiny_H3N2.pth',
        }
        
        model_file = models.get(virus_type)
        
        if not model_file:
            flash('Invalid virus model selection.')
            return redirect(request.url)

        try:
            # Initialize and load the selected model
            model = metafluad_model(model_file)  # Pass the selected model file to the model class
            
            # Use the model to predict distances with the user-defined batch size
            query_df = pd.read_csv(query_filename)
            reference_df = pd.read_csv(reference_filename)
            query_strains = query_df['strain'].tolist()
            reference_strain = reference_df['strain'].tolist()[0]

            distances = model.distances(query_filename, reference_filename, batch_size=batch_size)

            # Prepare predictions and save as CSV file
            predictions = [{'strain': strain, 'prediction': distance} for strain, distance in zip(query_strains, distances)]
            csv_file_path = os.path.join(app.config['UPLOAD_FOLDER'], "predictions.csv")
            with open(csv_file_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["strain", "prediction"])
                writer.writeheader()
                writer.writerows(predictions)

            # Render result page with conditionally displayed table
            return render_template(
                'result.html', 
                predictions=predictions, 
                reference_strain=reference_strain, 
                csv_file="predictions.csv", 
                csv_only=csv_only
            )
        
        except Exception as e:
            flash(f"An error occurred: {e}")
            return redirect(request.url)
        finally:
            # Clean up memory and delete files after processing
            if query_df is not None:
                del query_df
            if reference_df is not None:
                del reference_df
            if model is not None:
                del model
            os.remove(query_filename)
            os.remove(reference_filename)
            gc.collect()

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    """Route to download the generated CSV file."""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)