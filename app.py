import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pyngrok import ngrok
import subprocess
import os

# Define the Neural Network Architecture
class BandwidthAllocator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BandwidthAllocator, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Get the last time step
        return out

# Prepare the Dataset
def prepare_data():
    X = np.random.rand(1000, 10, 5)  # Example: 1000 samples, 10 time steps, 5 features
    y = np.random.rand(1000, 1)  # Example: 1000 samples, 1 target
    return torch.FloatTensor(X), torch.FloatTensor(y)

# Train the Model with Loss Tracking
def train_model(model, X_tensor, y_tensor, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if (epoch + 1) % 10 == 0:
            st.write(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    return loss_history

# Evaluate the Model
def evaluate_model(model):
    model.eval()
    with torch.no_grad():
        test_data = np.random.rand(10, 10, 5)
        test_tensor = torch.FloatTensor(test_data)
        predictions = model(test_tensor)
        return predictions.numpy()

# Plot Loss Curve
def plot_loss_curve(loss_history):
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    st.pyplot(plt)

# Streamlit Interface
def main():
    st.title("Interactive Neural Network: Bandwidth Allocation")
    st.write("This app trains a neural network to predict future bandwidth usage based on historical data.")

    # Sidebar for user inputs
    st.sidebar.header("Model Parameters")
    input_size = 5
    hidden_size = st.sidebar.slider("Hidden Layer Size", 16, 128, 64, step=16)
    output_size = 1
    num_epochs = st.sidebar.slider("Number of Epochs", 10, 200, 100, step=10)
    learning_rate = st.sidebar.selectbox("Learning Rate", [0.01, 0.001, 0.0001], index=1)

    st.sidebar.write("### Upload Custom Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

    if st.sidebar.button("Train Model"):
        # Load data
        if uploaded_file is not None:
            st.write("Custom dataset uploaded. Using uploaded data.")
            # Assuming the dataset has the required format
            data = np.genfromtxt(uploaded_file, delimiter=",", skip_header=1)
            X, y = data[:, :-1], data[:, -1:]
            X_tensor, y_tensor = torch.FloatTensor(X), torch.FloatTensor(y)
        else:
            st.write("No dataset uploaded. Using generated data.")
            X_tensor, y_tensor = prepare_data()

        # Train the model
        model = BandwidthAllocator(input_size, hidden_size, output_size)
        st.write("Training the model...")
        loss_history = train_model(model, X_tensor, y_tensor, num_epochs, learning_rate)

        st.write("Training complete!")
        plot_loss_curve(loss_history)

        # Evaluate the model
        predictions = evaluate_model(model)
        st.write("Sample Predictions (from random test data):")
        st.write(predictions)

# Ngrok and Streamlit Deployment Function
def deploy_app():
    # Set Ngrok authentication token
    # IMPORTANT: Replace with your actual Ngrok auth token
    NGROK_AUTH_TOKEN = 'your_ngrok_auth_token_here'
    
    # Set the auth token
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    
    # Create a public URL
    public_url = ngrok.connect(port=8501)
    print(f"Public URL: {public_url}")
    
    # Run the Streamlit app
    os.system('streamlit run your_script.py')

# Main execution
if __name__ == "__main__":
    # Choose between running the app directly or deploying with Ngrok
    # Uncomment the appropriate line based on your deployment needs
    
    # Option 1: Run locally
    main()
    
    # Option 2: Deploy with Ngrok (uncomment if needed)
    # deploy_app()