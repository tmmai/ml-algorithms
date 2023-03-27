import numpy as np
import unittest

from softmax_regression import SoftmaxRegression

class TestSoftmaxRegression(unittest.TestCase):
	_X = np.array([
		[1, 2, 3],
		[4, 5, 6]
		])

	_Y = np.array([
		[0, 0, 1],
		[0, 1, 0],
		[0, 0, 0],
		[1, 0, 0]
		])

	_W = np.array([
		[1, 2, 3, 4],
        [5, 6, 7, 8]
        ])

	_b = np.array([1, 2, 3, 4])

	_Z = np.array([
		[22, 28, 34],
		[28, 36, 44],
		[34, 44, 54],
		[40, 52, 64]
		])


	def test_linear(self):
		model = SoftmaxRegression()
		linear = model.linear(self._X, self._W, self._b)

		message = f'Failed-- \nresult: \n{linear}, \ntrue: \n{self._Z}'
		self.assertTrue((linear == self._Z).all(), message)


	def test_softmax(self):
		model = SoftmaxRegression()
		Z = model.linear(self._X, self._W, self._b)
		H = model.softmax(Z)

		col0_sum = np.sum(np.exp(self._Z[:, 0]))
		col1_sum = np.sum(np.exp(self._Z[:, 1]))
		col2_sum = np.sum(np.exp(self._Z[:, 2]))

		H_true = np.array([
			[np.exp(self._Z[0, 0]) / col0_sum, np.exp(self._Z[0, 1]) / col1_sum, np.exp(self._Z[0, 2]) / col2_sum],
			[np.exp(self._Z[1, 0]) / col0_sum, np.exp(self._Z[1, 1]) / col1_sum, np.exp(self._Z[1, 2]) / col2_sum],
			[np.exp(self._Z[2, 0]) / col0_sum, np.exp(self._Z[2, 1]) / col1_sum, np.exp(self._Z[2, 2]) / col2_sum],
			[np.exp(self._Z[3, 0]) / col0_sum, np.exp(self._Z[3, 1]) / col1_sum, np.exp(self._Z[3, 2]) / col2_sum]
			])

		message = f'Failed-- \nresult: \n{H}, \ntrue: \n{H_true}'
		self.assertTrue((H_true == H).all(), message)


	def test_compute_cost(self):
		model = SoftmaxRegression()
		Z = model.linear(self._X, self._W, self._b)
		H = model.softmax(Z)
		cost = model.compute_cost(H, self._Y)

		col0_sum = np.sum(np.exp(self._Z[:, 0]))
		col1_sum = np.sum(np.exp(self._Z[:, 1]))
		col2_sum = np.sum(np.exp(self._Z[:, 2]))

		cost_true = -1 * (np.log(np.exp(self._Z[3, 0]) / col0_sum) + np.log(np.exp(self._Z[1, 1]) / col1_sum)\
			+ np.log(np.exp(self._Z[0, 2]) / col2_sum))

		message = f'Failed-- result: {cost}, true: {cost_true}'
		self.assertEquals(cost, cost_true, message)


	def test_predict(self):
		model = SoftmaxRegression()
		W, b = model.fit(self._X, self._Y, alpha=0.01, epochs=10000)
		Y_hat = np.argmax(model.predict(self._X, W, b), axis=0).reshape(-1)
		Y_true = np.argmax(self._Y, axis=0).reshape(-1)


		message = f'Failed-- \nresult: \n{Y_hat}, \ntrue: \n{Y_true}'
		self.assertTrue((Y_hat == Y_true).all(), message)


if __name__ == '__main__':
	unittest.main()
