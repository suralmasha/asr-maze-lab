"""End-to-end integration checks for browser-based local training."""
import io
import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import wave
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import numpy as np

import lab_server
from training_web import TrainingManager


def tone(freq):
    times = np.arange(16000, dtype=float) / 16000
    samples = (0.28 * np.sin(2 * np.pi * freq * times) * np.hanning(16000) * 32767).astype('<i2')
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setframerate(16000); w.setnchannels(1); w.setsampwidth(2)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


class TestWebTraining(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.p_data = patch.object(lab_server, 'DATA', root / 'dataset')
        self.p_model = patch.object(lab_server, 'MODEL', root / 'models' / 'keyword_classifier.npz')
        self.p_training = patch.object(lab_server, 'training', TrainingManager())
        for p in (self.p_data, self.p_model, self.p_training): p.start()
        lab_server._classifier = None; lab_server._classifier_stamp = None
        for label in ('up', 'down', 'left', 'right'):
            (lab_server.DATA / label).mkdir(parents=True)
        self.http = ThreadingHTTPServer(('127.0.0.1', 0), lab_server.Handler)
        self.base = f'http://127.0.0.1:{self.http.server_port}'
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.http.shutdown(); self.http.server_close(); self.thread.join()
        for p in (self.p_training, self.p_model, self.p_data): p.stop()
        self.temp.cleanup()

    def request(self, path, body=None, mime='audio/wav'):
        req = urllib.request.Request(self.base + path, data=body,
            headers={'Content-Type': mime} if body is not None else {},
            method='POST' if body is not None else 'GET')
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as err:
            return err.code, err.read()

    def test_train_and_explain(self):
        code, msg = self.request('/api/train/start', b'', 'application/json')
        self.assertEqual(code, 400)
        self.assertIn('2 записи', json.loads(msg)['error'])
        for label, freq in [('up', 220), ('down', 380), ('left', 510), ('right', 730)]:
            for extra in (0, 11):
                status, data = self.request(f'/api/samples?label={label}', tone(freq + extra))
                self.assertEqual(status, 201, data)
        status, started = self.request('/api/train/start', b'', 'application/json')
        self.assertEqual(status, 202, started)
        self.assertEqual(json.loads(started)['state'], 'running')
        deadline = time.monotonic() + 15
        while True:
            status, raw = self.request('/api/train/status')
            data = json.loads(raw)
            self.assertEqual(status, 200)
            if data['state'] != 'running': break
            self.assertLess(time.monotonic(), deadline, data)
            time.sleep(0.08)
        self.assertEqual(data['state'], 'done', data)
        self.assertEqual(data['percent'], 100)
        self.assertTrue(data['ready'])
        self.assertTrue(lab_server.MODEL.exists())
        with np.load(lab_server.MODEL, allow_pickle=False) as values:
            self.assertEqual(list(values['train_names'])[0], 'up/01.wav')
        status, image = self.request('/api/train/plot')
        self.assertEqual(status, 200)
        self.assertTrue(image.startswith(b'\x89PNG'))
        status, result = self.request('/api/train/probe', tone(220))
        self.assertEqual(status, 200)
        data = json.loads(result)
        self.assertEqual(data['label'], 'up')
        self.assertEqual(len(data['neighbors']), 3)
        self.assertEqual(data['neighbors'][0]['file'], 'up/01.wav')
        status, image = self.request('/api/train/plot')
        self.assertTrue(image.startswith(b'\x89PNG'))
        code, html = self.request('/train.html')
        self.assertEqual(code, 200)
        self.assertIn(b'train.js', html)
        code, script = self.request('/train.js')
        self.assertEqual(code, 200)
        self.assertIn(b'/api/train/start', script)
        status, retrained = self.request('/api/train/start', b'', 'application/json')
        self.assertEqual(status, 202)

if __name__ == '__main__': unittest.main()
