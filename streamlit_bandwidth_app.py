import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import logging

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

# Bandwidth Service Class
class BandwidthService:
    def __init__(self, model_path, data_path=None):
        self.model_path = model_path
        self.data_path = data_path
        
        try:
            self.model = joblib.load(model_path)
            st.sidebar.success("Pre-trained model loaded successfully!")
        except FileNotFoundError:
            st.sidebar.error(f"Model file not found at {model_path}.")
            self.model = None

    def allocate_bandwidth(self, data):
        if self.model is None:
            st.error("Model not loaded. Cannot allocate bandwidth.")
            return None

        # Convert categorical features to numeric (if necessary)
        if "Activity_Type" in data.columns:
            data["Activity_Type"] = data["Activity_Type"].astype("category").cat.codes

        predictions = self.model.predict(data)
        return predictions

# Streamlit Interface
def main():
    st.title("Interactive Neural Network & Bandwidth Service")
    st.write("This app allows you to train a neural network or use a pre-trained bandwidth allocation model.")

    # Sidebar for mode selection
    mode = st.sidebar.selectbox("Choose Mode", ["Neural Network Training", "Bandwidth Service"])

    if mode == "Neural Network Training":
        # Neural Network Training Interface
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

    elif mode == "Bandwidth Service":
        # Bandwidth Service Interface
        st.sidebar.header("Pre-trained Model")
        model_path = "C:\\Users\\mianm\\Desktop\\code\\bandwidth-allocator\\Model\\Network\\models\\bandwidth_model.pkl"
        data_file = st.sidebar.file_uploader("Upload data file (CSV)", type=["csv"])

        if data_file:
            service = BandwidthService(model_path=model_path)

            # Load data
            try:
                data = pd.read_csv(data_file)
                st.write("Data loaded successfully:")
                st.dataframe(data.head())

                # Allocate bandwidth
                predictions = service.allocate_bandwidth(data)
                if predictions is not None:
                    data["Priority"] = predictions
                    st.write("Predictions:")
                    st.dataframe(data)

                    # Option to download results
                    st.download_button(
                        "Download Predictions", 
                        data.to_csv(index=False), 
                        file_name="predictions.csv"
                    )
            except Exception as e:
                st.error(f"Error loading data: {e}")

if __name__ == "__main__":
    main()
