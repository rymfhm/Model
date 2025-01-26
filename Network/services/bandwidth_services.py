import pandas as pd
import joblib  # To load the trained model
import logging
from config import Config

logger = logging.getLogger(__name__)

class BandwidthService:
    """
    Service for optimizing bandwidth allocation using a trained ML model.
    """

    def __init__(self, model_path="C:\\Users\\Admin\\Documents\\Network\\Network-lifecyle-management-system\\models\\bandwidth_model.pkl", data_path="data/bandwidth_data.csv"):
        """
        Initialize the service with the trained model and input data.
        :param model_path: Path to the trained ML model file.
        :param data_path: Path to the CSV file containing input data.
        """
        self.model_path = model_path
        self.data_path = data_path
        
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded successfully from {model_path}.")
        except FileNotFoundError:
            logger.error(f"Model file not found at {model_path}. Please ensure it exists.")
            self.model = None
        
        try:
            self.bandwidth_data = pd.read_csv(data_path)
            logger.info(f"Input data loaded successfully from {data_path}.")
        except FileNotFoundError:
            logger.error(f"Data file not found at {data_path}. Please ensure it exists.")
            self.bandwidth_data = None

    def allocate_bandwidth(self):
        """
        Use the trained model to predict bandwidth allocation priorities.
        """
        if self.model is None or self.bandwidth_data is None:
            logger.error("Model or input data is unavailable. Bandwidth allocation cannot proceed.")
            return None
        
        # Prepare features for the model
        feature_columns = ["Activity_Type", "Bandwidth_Usage_Mbps"]  # Modify as per your model's requirements
        if not all(col in self.bandwidth_data.columns for col in feature_columns):
            logger.error("Input data does not contain required features for prediction.")
            return None
        
        # Convert categorical features to numeric (if necessary)
        # Assuming 'Activity_Type' was encoded during training
        self.bandwidth_data["Activity_Type"] = self.bandwidth_data["Activity_Type"].astype("category").cat.codes
        
        # Perform predictions
        predictions = self.model.predict(self.bandwidth_data[feature_columns])
        self.bandwidth_data["Priority"] = predictions
        
        logger.info("Bandwidth allocation priorities predicted successfully.")
        return self.bandwidth_data

    def save_allocations(self, output_path="data/prioritized_bandwidth.csv"):
        """
        Save the predicted bandwidth allocations to a CSV file.
        :param output_path: Path where the predictions will be saved.
        """
        allocations = self.allocate_bandwidth()
        if allocations is not None:
            allocations.to_csv(output_path, index=False)
            logger.info(f"Predicted bandwidth allocations saved to {output_path}.")
        else:
            logger.error("Failed to save predicted bandwidth allocations.")

# Usage example
if __name__ == "__main__":
    service = BandwidthService()
    service.save_allocations()
