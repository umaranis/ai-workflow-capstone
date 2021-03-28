# !/usr/bin/env python
from datetime import date
import os
import sys
import unittest
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', "src"))
## import model specific functions and variables
from logger import update_train_log, update_predict_log

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'test', 'logs')


class LoggingTest(unittest.TestCase):
    """
    Testing logging
    """

    def test_01_train(self):
        """
        ensure log file is created
        """
        today = date.today()
        log_file = os.path.join(LOG_DIR, "train-{}-{}.log".format(today.year, today.month))
        if os.path.exists(log_file):
            os.remove(log_file)

        ## update the log
        date_range = ('2017-11-29', '2019-05-24')
        runtime = "00:00:01"
        model_version = 0.1
        model_version_note = "test model"

        update_train_log(date_range, runtime,
                         model_version, model_version_note, test=False)

        self.assertTrue(os.path.exists(log_file))

    def test_02_train(self):
        """
        ensure that content can be retrieved from log file
        """
        today = date.today()
        log_file = os.path.join(LOG_DIR, "train-{}-{}.log".format(today.year, today.month))

        ## update the log
        date_range = ('2017-11-29', '2019-05-24')
        runtime = "00:00:01"
        model_version = 0.1
        model_version_note = "test model"

        update_train_log(date_range, runtime,
                         model_version, model_version_note, test=False)

        df = pd.read_csv(log_file)
        logged_model_version = df["model_version"].iloc[-1]
        self.assertEqual(model_version, logged_model_version)

    def test_03_predict(self):
        """
        ensure log file is created
        """
        today = date.today()
        log_file = os.path.join(LOG_DIR, "predict-{}-{}.log".format(today.year, today.month))
        if os.path.exists(log_file):
            os.remove(log_file)

        ## update the log
        y_pred = [0]
        runtime = "00:00:02"
        model_version = 0.1

        update_predict_log(y_pred, runtime,
                           model_version, None, test=False)

        self.assertTrue(os.path.exists(log_file))

    def test_04_predict(self):
        """
        ensure that content can be retrieved from log file
        """
        today = date.today()
        log_file = os.path.join(LOG_DIR, "predict-{}-{}.log".format(today.year, today.month))

        ## update the log
        y_pred = [0]
        runtime = "00:00:02"
        model_version = 0.1

        update_predict_log(y_pred, runtime,
                           model_version, None, test=False)

        df = pd.read_csv(log_file)
        logged_y_pred = df['y_pred'].iloc[-1]
        self.assertEqual(str(y_pred), logged_y_pred)


### Run the tests
if __name__ == '__main__':
    unittest.main()

