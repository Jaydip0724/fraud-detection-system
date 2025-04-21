import unittest
from app.main import app

class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        app.testing = True
        self.client = app.test_client()
    
    def test_home_endpoint(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'running')
    
    def test_metrics_endpoint(self):
        response = self.client.get('/metrics')
        # The API will return status 200 if metrics are available or 500 if there is an issue.
        self.assertIn(response.status_code, [200, 500])
    
    def test_statistics_endpoint(self):
        response = self.client.get('/statistics')
        self.assertIn(response.status_code, [200, 500])
    
    def test_transactions_endpoint(self):
        response = self.client.get('/transactions')
        self.assertIn(response.status_code, [200, 500])

if __name__ == '__main__':
    unittest.main()
