import unittest
import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from src.descriptive_statistics import summarize_categorical, describe_numerical

class TestDescriptiveStatistics(unittest.TestCase):
    def test_describe_numerical(self):
        # Create a small test dataframe
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [2.0, 3.0, 4.0, np.nan, 5.0],
            'C': ['a', 'b', 'c', 'd', 'e']
        })

        # Call the function with the test dataframe
        result = describe_numerical(df)

        # Check the result
        expected_result = pd.DataFrame({
            'A': [5, 3.0, 1.5811388300841898, 1.0, 2.0, 3.0, 4.0, 5.0],
            'B': [4, 3.5, 1.29099445, 2.0, 2.75, 3.5, 4.25, 5.0]
        }, index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])

        pd.testing.assert_frame_equal(result, expected_result)
    
    def test_summarize_categorical(self):
        # Test data for categorical columns
        data = {'CategoryColumn': ['A', 'B', 'A', 'C', 'B']}
        df = pd.DataFrame(data)
        
        expected_output = {'CategoryColumn': {'A': 2, 'B': 2, 'C': 1}}
        actual_output = summarize_categorical(df)

        # Convert dictionary values to sets before comparison
        expected_output_set = {k: set(v.items()) for k, v in expected_output.items()}
        actual_output_set = {k: set(v.items()) for k, v in actual_output.items()}
        
        self.assertEqual(expected_output_set, actual_output_set)
        
if __name__ == '__main__':
    unittest.main()
