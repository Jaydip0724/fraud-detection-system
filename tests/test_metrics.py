import unittest
import numpy as np
from app.utils.metrics import calculate_metrics

class TestMetrics(unittest.TestCase):
    def test_calculate_metrics(self):
        # Prepare example true and predicted labels
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1])
        precision, recall, f1, accuracy = calculate_metrics(y_true, y_pred)
        
        # Expected results:
        # - For precision: only one predicted positive (correct) so precision = 1.0.
        # - For recall: one true positive detected out of two actual positives -> recall = 0.5.
        # - f1 score is computed as 2*(precision*recall)/(precision+recall)
        # - Accuracy: 3/4 = 0.75.
        self.assertAlmostEqual(precision, 1.0)
        self.assertAlmostEqual(recall, 0.5)
        expected_f1 = 2 * (1.0 * 0.5) / (1.0 + 0.5)
        self.assertAlmostEqual(f1, expected_f1)
        self.assertAlmostEqual(accuracy, 0.75)

if __name__ == '__main__':
    unittest.main()
