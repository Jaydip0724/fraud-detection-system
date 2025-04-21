import unittest
import pandas as pd
from app.utils.data_processing import clean_data

class TestDataProcessing(unittest.TestCase):
    def test_clean_data(self):
        # Create a simple DataFrame with some NaN values.
        df = pd.DataFrame({
            'a': [1, 2, None, 4],
            'b': [None, 2, 3, 4]
        })
        # Assume that clean_data drops rows with any NaN values.
        cleaned_df = clean_data(df)
        
        # After cleaning, no NaN values should remain.
        self.assertFalse(cleaned_df.isnull().values.any())
        # In this case, only two rows have complete data.
        self.assertEqual(len(cleaned_df), 2)

if __name__ == '__main__':
    unittest.main()
