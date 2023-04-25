from sklearn.cluster import KMeans, DBSCAN
import numpy as np
import time
from collections import Counter

from abc import abstractmethod, ABC

class TeamClassificator(ABC):
    
    @abstractmethod
    def classify_persons_into_categories(self, detections: np.ndarray, max_height = -1, max_width = -1):
        pass
    
    def preprocess_detections(self, detections: np.ndarray):
        return detections.reshape(detections.shape[0], -1)

    def pad_detections(self, bboxes: np.ndarray, max_width: int, max_height: int):
        padded_bboxes = []
        # print(f"Max {max_height} {max_width}")
        for i, bbox in enumerate(bboxes):
            lpad, upad = int((max_width - bbox.shape[1]) / 2), int((max_height - bbox.shape[0]) / 2)
            rpad, dpad = lpad + int((max_width - bbox.shape[1]) % 2), upad + int((max_height - bbox.shape[0]) % 2)
            # print(f"bbox shape: {bbox.shape}")
            # print(f"l, u, r, d: {lpad} {upad} {rpad} {dpad}")
            padded_bboxes.append(np.pad(bbox, ((upad, dpad), (lpad, rpad), (0, 0)), mode="mean"))
            # print(f"After padding shape: {padded_bboxes[i].shape}")
        padded_bboxes = np.array(padded_bboxes)
        # print(f"All shape: {padded_bboxes.shape}")
        return padded_bboxes

class DBSCANTeamClassificator(TeamClassificator):
    
    def __init__(self) -> None:
        self.alg = DBSCAN(eps=3, min_samples=2)
    
    def classify_persons_into_categories(self, detections: np.ndarray, max_height=-1, max_width=-1):
        start_time = time.time()
        labels = self.alg.fit_predict(self.preprocess_detections(detections))
        # print(f"Labels: {labels}")
        # print(f"Fit and predict took: {time.time() - start_time:.2f}s")
        return labels
    
class KMeansTeamClassificator(TeamClassificator):
    
    def __init__(self) -> None:
        self.alg = KMeans(n_clusters=3)
    
    def classify_persons_into_categories(self, detections: np.ndarray, max_height = -1, max_width = -1):
        """Tries to classify objects into two teams and a referee using clustering algorithms.

        Args:
            detections (np.ndarray): N*H*W*C 
            max_height (int): _description_
            max_width (int): _description_

        Returns:
            _type_: _description_
        """
        start_time = time.time()
        # Detections
        # if self.detections is None:
        #     self.detections = detections
        # else:
        #     self.detections = np.vstack((self.detections, detections))
        # print(f"Shape of detections: {detections.shape} {self.detections.shape}")
        # Transfer because you will need it for padding
        # if max_height > self.max_height:
        #     self.max_height = max_height
        # if max_width > self.max_width:
        #     self.max_width = max_width
        detections_ = self.preprocess_detections(detections)
        self.alg.fit(detections_)
        labels = self.alg.predict(detections_)
        # print(f"Labels: {labels}")
        # print(f"Fit and predict took: {time.time() - start_time:.2f}s")
        return labels
        
    