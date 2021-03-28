#!/usr/bin/env python
import os
import unittest
import requests
import re
from ast import literal_eval

port = 8080

try:
    requests.post('http://localhost:{}/ping'.format(port))
    server_available = True
except:
    server_available = False


## test class for the main window function
class ApiTest(unittest.TestCase):
    """
    Test model API endpoints
    """

    @unittest.skipUnless(server_available, "local server is not running")
    def test_01_train(self):
        """
        test the train functionality
        """

        request_json = {'mode': 'test'}
        r = requests.post('http://localhost:{}/train'.format(port), json=request_json)
        train_complete = re.sub("\W+", "", r.text)
        self.assertEqual(train_complete, 'true')

    @unittest.skipUnless(server_available, "local server is not running")
    def test_02_predict_empty(self):
        """
        ensure appropriate failure types
        """

        ## provide no data at all
        r = requests.post('http://localhost:{}/predict'.format(port))
        self.assertEqual(re.sub('\n|"', '', r.text), "[]")

        ## provide improperly formatted data
        r = requests.post('http://localhost:{}/predict'.format(port), json={"key": "value"})
        self.assertEqual(re.sub('\n|"', '', r.text), "[]")

    @unittest.skipUnless(server_available, "local server is not running")
    def test_03_predict(self):
        """
        test the predict functionality
        """

        query_data = {'country': 'United Kingdom',
                      'date': '2018-06-12'}

        query_type = 'dict'
        request_json = {'query': query_data, 'type': query_type, 'mode': 'test'}

        r = requests.post('http://localhost:{}/predict'.format(port), json=request_json)
        print(r.text)
        response = literal_eval(r.text)

        for p in response['United Kingdom']['y_pred']:
            self.assertTrue(p is not None)

    @unittest.skipUnless(server_available, "local server is not running")
    def test_04_predict_all(self):
        """
        test the predict functionality
        """
        query_data = {'country': 'all',
                      'date': '2018-06-12'}

        query_type = 'dict'
        request_json = {'query': query_data, 'type': query_type, 'mode': 'test'}

        r = requests.post('http://localhost:{}/predict'.format(port), json=request_json)
        print(r.text)
        response = literal_eval(r.text)

        for p in ['United Kingdom', 'Portugal']:
            self.assertTrue(response[p]['y_pred'] is not None)

    @unittest.skipUnless(server_available, "local server is not running")
    def test_05_predict_multiple(self):
        """
        test the predict functionality
        """
        query_data = {'country': 'United Kingdom,Portugal',
                      'date': '2018-06-12'}

        query_type = 'dict'
        request_json = {'query': query_data, 'type': query_type, 'mode': 'test'}

        r = requests.post('http://localhost:{}/predict'.format(port), json=request_json)
        print(r.text)
        response = literal_eval(r.text)

        for p in ['United Kingdom', 'Portugal']:
            self.assertTrue(response[p]['y_pred'] is not None)

    @unittest.skipUnless(server_available, "local server is not running")
    def test_06_logs(self):
        """
        test the log functionality
        """

        file_name = 'test-train-2020-6.log'
        r = requests.get('http://localhost:{}/logs/{}'.format(port, file_name))

        with open(file_name, 'wb') as f:
            f.write(r.content)

        self.assertTrue(os.path.exists(file_name))

        if os.path.exists(file_name):
            os.remove(file_name)


### Run the tests
if __name__ == '__main__':
    unittest.main()