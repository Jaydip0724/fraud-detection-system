import unittest
import numpy as np
from app.utils.model import train_model, predict

class TestModel(unittest.TestCase):
    def setUp(self):
        # Prepare a dummy dataset.
        # Assuming your features include 28 'V' values and 1 'amount' => total 29 columns.
        np.random.seed(42)
        self.X_train = np.random.rand(100, 29)
        self.y_train = np.random.choice([0, 1], size=100)
        self.X_test = np.random.rand(10, 29)
    
    def test_train_and_predict(self):
        # Train the model using the training dataset.
        model, scaler = train_model(self.X_train, self.y_train)
        
        # Generate predictions on the test dataset.
        predictions = predict(model, scaler, self.X_test)
        self.assertEqual(len(predictions), self.X_test.shape[0])
        
        # Ensure that each prediction is either 0 or 1.
        for pred in predictions:
            self.assertIn(pred, [0, 1])

if __name__ == '__main__':
    unittest.main()
