import numpy as np
from ProjectFolder.Code.evaluate import pitch_confusion, final_score

x = np.random.binomial(1, 0.005, (300, 500))
y = np.random.binomial(1, 0.003, (300, 500))

pitch_confusion(x, y, vtype='joint', save_path='test.png', description="3")
