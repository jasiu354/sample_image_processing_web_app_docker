import cv2
import os
import unittest
from backend.opencv import detect_faces
import numpy as np

class TestDetectFacesMethod(unittest.TestCase):

    def test_detect_faces(self):
        
        a_face = cv2.imdecode(np.fromfile(os.path.join(os.path.dirname(__file__), 'resources/a_face.jpg'), np.uint8), cv2.IMREAD_UNCHANGED)
        not_a_face = cv2.imdecode(np.fromfile(os.path.join(os.path.dirname(__file__), 'resources/not_a_face.jpg'), np.uint8), cv2.IMREAD_UNCHANGED)
        
        self.assertEqual(len(detect_faces(a_face)), 1), "Should be 1, found a face"
        self.assertEqual(len(detect_faces(not_a_face)), 0), "Should be 0, did not find a face"


if __name__ == '__main__':
    unittest.main()