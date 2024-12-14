# House Price Prediction using Linear Regression

This project is a Python-based implementation of a linear regression model to predict house prices based on their size. The code uses the `scikit-learn` library for training the model and `matplotlib` for generating visualizations. The project also includes functionality for making predictions and saving the resulting graph to an image file.

## Project Overview

- **Objective**: Predict the price of a house based on its size using a linear regression model.
- **Technology Used**:
  - `Python 3.12.3`
  - `scikit-learn` for machine learning
  - `pandas` for data manipulation
  - `matplotlib` for creating visualizations

## Features

1. **Train a Linear Regression Model**: The model is trained using historical data of house sizes and their prices.
2. **Predict House Price**: The user can input the size of a house, and the model will predict its price.
3. **Graph Visualization**: A scatter plot is generated showing house sizes and prices, with a regression line to visualize the linear relationship between them.
4. **Export Graph**: The generated graph is saved as `output.png` in the same directory.
5. **User Input**: The script prompts the user for input to predict house prices based on a given size.

## Prerequisites

To run the code, you need to install the following Python libraries:

- `numpy` - for numerical operations
- `pandas` - for data manipulation
- `matplotlib` - for creating visualizations
- `scikit-learn` - for implementing linear regression

You need to be using **Python 3.12.3** (or a compatible version) to run this project successfully.

### To install dependencies, use:

```bash
pip install numpy pandas matplotlib scikit-learn
