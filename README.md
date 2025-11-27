House Price Prediction with Images and Tabular Data
This project builds and evaluates deep learning models to predict residential property prices using both property images and structured tabular features. The workflow combines computer vision with classic regression features such as building size, land size, number of bedrooms, and location.

Project Overview
Loads a CSV of property listings and a folder of corresponding property images.

Matches each row in the CSV to an image file and filters to records with valid images.

Builds a dataset that merges image data with cleaned and scaled numeric and categorical features.

Trains convolutional neural network (CNN) models using PyTorch to predict house price from the combined inputs.

Evaluates model performance using metrics such as MAE, MSE, and 
R
2
R 
2
 .

Data
Tabular data file (e.g. property_final.csv) containing:

price(USD) (target)

building_area(m²)

land_area(m²)

bedrooms

location and other metadata

Image folder (e.g. property_images/) with one image per property, matched by an id field.

The notebook filters to rows with existing images and creates a final dataset of matched samples.

Methods
Environment: Python, PyTorch, torchvision, scikit-learn, pandas, NumPy, Matplotlib, and tqdm.

Device configuration automatically selects GPU (cuda) if available and applies basic CUDA optimizations.

A custom HousePriceImageDataset class:

Handles image loading and transformation.

Fills missing numeric values and encodes location with LabelEncoder.

Standardizes numeric and encoded features with StandardScaler.

Target prices are scaled with StandardScaler before training, then inverse-transformed for evaluation.

Data is split into train, validation, and test sets (e.g. 70/15/15).

Training and Evaluation
CNN backbones from torchvision.models are used as feature extractors for the image branch.

Tabular features are concatenated with image embeddings and passed through fully connected layers for regression.

Training loops track:

Training and validation loss

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

R
2
R 
2
  score

Final metrics on the validation and test sets are reported in the notebook, along with basic plots of loss and metric curves.

How to Run
Clone this repository and set up a Python environment with the required packages (PyTorch, torchvision, scikit-learn, pandas, NumPy, Matplotlib, tqdm).

Place property_final.csv and the property_images/ folder in the paths expected by the notebook (or update the paths in the data loading cell).

Open Mini-Project-Tapiwa-Mhondiwa.ipynb in Jupyter or VS Code.

Run the cells in order to:

Configure the device and environment

Load and match data

Build datasets and dataloaders

Train and evaluate the model

Repository Structure
Mini-Project-Tapiwa-Mhondiwa.ipynb – main notebook with all code and experiments.

property_final.csv – property metadata and target prices (not included in this repo by default).

property_images/ – folder containing property images (not included in this repo by default).

Future Improvements
Experiment with different CNN architectures and hyperparameters.

Add more tabular features (e.g. neighborhood statistics, year built).

Implement cross-validation and more robust model comparison.

Export trained models and create an API or simple web app for serving predictions.

