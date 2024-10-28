[![Render](https://img.shields.io/badge/OnRender-Deployed-9b59b6?logo=cloud&logoColor=white)](https://smartad-latest.onrender.com)
[![Docker](https://img.shields.io/badge/Docker-Pull%20Image-1abc9c?logo=docker&logoColor=white)](https://hub.docker.com/r/ameknas/smartad)

# SmartAD: Predicting Antigenic Distances of Influenza Viruses

---

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Available Models](#available-models)
5. [How SmartAD Works](#how-smartad-works)
6. [Data Input Format](#data-input-format)
7. [Contributing](#contributing)
8. [License](#license)

---

## Introduction

Antigenic distances represent a quantitative measure of how much the immune system recognizes one viral strain as being distinct from another. For influenza viruses, understanding antigenic distances is crucial because it helps predict how well the immune system might respond to a new strain based on previous exposure or vaccination. Accurate antigenic distance predictions can inform vaccine strain selection and predict vaccine efficacy across different influenza seasons.

Influenza viruses, especially types A and B, undergo constant mutations in surface proteins like hemagglutinin (HA). These mutations can cause the virus to "drift" antigenically, meaning that immunity from prior infection or vaccination may be less effective against new strains. This drift is the main reason why the flu vaccine must be updated regularly. Therefore, predicting antigenic distances between viral strains is vital in the battle against influenza.

**SmartAD**, hosted as a web tool on **OnRender**, fits into this landscape by providing a streamlined interface for predicting antigenic distances between influenza strains. The predictions are powered by models trained on extensive datasets, building upon the existing methodology of a tool called **MetaFluAD**.

### MetaFluAD: The Basis of SmartAD

MetaFluAD, a cutting-edge tool developed by Qitao Jia, Yuanling Xia, Fanglin Dong, and Weihua Li, is a meta-learning framework designed to predict antigenic distances between influenza viruses. The method was originally published in a [paper](https://academic.oup.com/bib/article/25/5/bbae395/7731492) titled *MetaFluAD: Meta-learning for Predicting Antigenic Distances among Influenza Viruses*, and the source code is available on GitHub [here](https://github.com/kpollop/metafluad).

**SmartAD** builds upon this work by using an enhanced version of the MetaFluAD models. The difference lies in the training data: while the original MetaFluAD models were trained on publicly available datasets, the models powering SmartAD have been trained on a more extensive and proprietary dataset maintained by the **Influenza, Respiratory Viruses and Cornoaviruses (IRVC)** section at the **National Microbiology Laboratory (NML)**. This larger dataset provides a deeper and broader understanding of antigenic distances across a wider variety of influenza strains, leading to more accurate predictions.

In essence, SmartAD leverages the power of machine learning models to help researchers and public health officials understand how different influenza strains might interact with human immunity, guiding better vaccine formulation and public health strategies.

---

## Installation

### Docker Image

#### Command Line

To install and run **SmartAD** via Docker using the command line, follow these steps:

1. **Pull the Docker Image**
   ```bash
   docker pull ameknas/smartad:latest
   ```
   
2. **Run the Docker Container**
   ```bash
   docker run -p 5000:5000 ameknas/smartad:latest
   ```
   
3. **Access the Application**
   - Open your web browser and navigate to:
     ```
     http://localhost:5000
     ```
   - This will open the **SmartAD** web interface where you can begin using the application.

### Docker Desktop

For users with Docker Desktop, you can pull and run the **SmartAD** container directly from the Docker Desktop interface:

1. **Open Docker Desktop**  
   *(Placeholder: Insert image of Docker Desktop main screen)*

2. **Search for the SmartAD Image**
   - In the **Images** tab, locate the "Search for images on Docker Hub" field.
   - Type `ameknas/smartad` to find the SmartAD image.
   - *(Placeholder: Insert image showing Docker Hub search field and locating SmartAD image)*

3. **Pull the Image**
   - Select the **Pull** option to download the latest image to your Docker Desktop.
   - *(Placeholder: Insert image of Docker Desktop’s pull option)*

4. **Run the Container**
   - After the image has been pulled, locate it under the **Images** tab in Docker Desktop.
   - Click **Run** next to the image name.
   - Configure the **port mapping** to `5000:5000` so you can access the application on `localhost:5000`.
   - *(Placeholder: Insert image of container configuration with port mapping)*

5. **Access the Application**
   - Open your web browser and navigate to:
     ```
     http://localhost:5000
     ```
   - This will take you to the **SmartAD** web interface, where you can begin using the application.

*(Placeholder: Insert image of SmartAD app interface in browser)*

By following these steps, you can easily install and run **SmartAD** using Docker, either through the command line or Docker Desktop, and access it directly from your browser.
