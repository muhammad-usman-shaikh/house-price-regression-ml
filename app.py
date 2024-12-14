# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LinearRegression

# # Step 1: Read data from CSV
# def load_data(file_path='data.csv'):
#     try:
#         data = pd.read_csv(file_path)
#         return data['house_size'].values, data['price'].values
#     except FileNotFoundError:
#         print(f"Error: File '{file_path}' not found. Ensure it's in the same directory.")
#         exit()

# # Step 2: Perform Linear Regression
# def train_model(house_sizes, house_prices):
#     X = house_sizes.reshape(-1, 1)
#     y = house_prices
#     model = LinearRegression()
#     model.fit(X, y)
#     return model

# # Step 3: Predict based on user input
# def predict_price(model, house_size):
#     return model.predict([[house_size]])[0]

# # Main program
# if __name__ == '__main__':
#     # Load data
#     house_sizes, house_prices = load_data()

#     # Train model
#     model = train_model(np.array(house_sizes), np.array(house_prices))
#     print("Model trained successfully!")

#     # Get user input
#     try:
#         house_size = float(input("Enter the house size (in sqft): "))
#         predicted_price = predict_price(model, house_size)
#         print(f"The predicted price for a house of size {house_size} sqft is: PKR {predicted_price:,.0f}")
#     except ValueError:
#         print("Invalid input. Please enter a numeric value for house size.")



# V2 works fine but y axis shows 1 2 3 and not in actual crore values
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression

# # Step 1: Read data from CSV
# def load_data(file_path='data.csv'):
#     try:
#         data = pd.read_csv(file_path)
#         return data['house_size'].values, data['price'].values
#     except FileNotFoundError:
#         print(f"Error: File '{file_path}' not found. Ensure it's in the same directory.")
#         exit()

# # Step 2: Perform Linear Regression
# def train_model(house_sizes, house_prices):
#     X = house_sizes.reshape(-1, 1)
#     y = house_prices
#     model = LinearRegression()
#     model.fit(X, y)
#     return model

# # Step 3: Predict based on user input
# def predict_price(model, house_size):
#     return model.predict([[house_size]])[0]

# # Step 4: Plot the data and the regression line
# def plot_graph(house_sizes, house_prices, model):
#     plt.scatter(house_sizes, house_prices, color='blue', label='Data points')
#     plt.plot(house_sizes, model.predict(house_sizes.reshape(-1, 1)), color='red', label='Regression line')
#     plt.title('Linear Regression - House Size vs Price')
#     plt.xlabel('House Size (sqft)')
#     plt.ylabel('Price ($)')
#     plt.legend()
#     plt.savefig('output.png')  # Save the plot as an image
#     print("Graph saved as 'output.png'")


# # Main program
# if __name__ == '__main__':
#     # Load data
#     house_sizes, house_prices = load_data()

#     # Train model
#     model = train_model(np.array(house_sizes), np.array(house_prices))
#     print("Model trained successfully!")

#     # Plot the graph
#     plot_graph(np.array(house_sizes), np.array(house_prices), model)

#     # Get user input
#     try:
#         house_size = float(input("Enter the house size (in sqft): "))
#         predicted_price = predict_price(model, house_size)
#         print(f"The predicted price for a house of size {house_size} sqft is: ${predicted_price:,.2f}")
#     except ValueError:
#         print("Invalid input. Please enter a numeric value for house size.")


# v3 test
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import FuncFormatter

# Step 1: Read data from CSV
def load_data(file_path='data.csv'):
    try:
        data = pd.read_csv(file_path)
        return data['house_size'].values, data['price'].values
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Ensure it's in the same directory.")
        exit()

# Step 2: Perform Linear Regression
def train_model(house_sizes, house_prices):
    X = house_sizes.reshape(-1, 1)
    y = house_prices
    model = LinearRegression()
    model.fit(X, y)
    return model

# Step 3: Predict based on user input
def predict_price(model, house_size):
    return model.predict([[house_size]])[0]

# Step 4: Plot the data and the regression line
def plot_graph(house_sizes, house_prices, model):
    plt.scatter(house_sizes, house_prices, color='blue', label='Data points')
    plt.plot(house_sizes, model.predict(house_sizes.reshape(-1, 1)), color='red', label='Regression line')
    
    # Customizing y-axis labels
    def format_price(x, pos):
        return f"${x:,.0f}"  # Format as a dollar amount with commas and no decimals

    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_price))
    
    # Adjusting the y-axis limits to ensure the entire range is visible
    plt.ylim(min(house_prices) - (0.1 * min(house_prices)), max(house_prices) + (0.1 * max(house_prices)))

    # Ensuring that everything fits without cutting off labels
    plt.title('Linear Regression - House Size vs Price')
    plt.xlabel('House Size (sqft)')
    plt.ylabel('Price ($)')
    plt.legend()
    
    # Adjust layout to avoid cutting off any content
    plt.tight_layout()
    
    plt.savefig('output.png')  # Save the plot as an image
    print("Graph saved as 'output.png'")

# Main program
if __name__ == '__main__':
    # Load data
    house_sizes, house_prices = load_data()

    # Train model
    model = train_model(np.array(house_sizes), np.array(house_prices))
    print("Model trained successfully!")

    # Plot the graph
    plot_graph(np.array(house_sizes), np.array(house_prices), model)

    # Get user input
    try:
        house_size = float(input("Enter the house size (in sqft): "))
        predicted_price = predict_price(model, house_size)
        print(f"The predicted price for a house of size {house_size} sq yard is: PKR {predicted_price:,.0f}")
    except ValueError:
        print("Invalid input. Please enter a numeric value for house size.")
