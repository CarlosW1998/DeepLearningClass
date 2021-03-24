from LogisticRegression import LogisticRegression
import numpy as np

ts_input = np.array([[0,0,1,0],
                    [1,1,1,0],
                    [1,0,1,1],
                    [0,1,1,1],
                    [0,1,0,1],
                    [1,1,1,1],
                    [0,0,0,0]])
ts_output = np.array([[0,1,1,0,0,1,0]]).T

model = LogisticRegression(4, 10)

model.fit(ts_input, ts_output, 1000)

print(model.weight)