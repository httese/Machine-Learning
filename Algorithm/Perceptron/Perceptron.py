#!/usr/bin/python
#encoding=utf-8

"""
A Python implementation of the Perceptron algorithm.
"""

import sys
from optparse import OptionParser

class Perceptron:
	# __doc__
	"""Perceptron Algorithm Model"""

	# class variable shared by all instances
	kind = 'Perceptron'

	# instance variable unique to each instance
	def __init__(self, step):
		self.train_set = []
		self.train_history = []

		self.w = []
		self.b = 0
		self.step = step

	def InputData(self, fl_path):
		"""
		input data from target file

		:param fl_path:
			path of the target file

		inport data fmt:
			1 character_1_value character_2_value ... character_n_value
			-1 character_1_value character_2_value ... character_n_value
			...
		"""

		self.train_set = []
		with open(fl_path,'r') as fp:
			for i in fp:
				i = i.replace('\r','').replace('\n','')
				if i != '':
					data = []
					for item in i.split():
						data.append(float(item))

					self.train_set.append((data[0], data[1:]))

		self.w = [0] * len(self.train_set[0][1])
		self.b = 0
		return self.train_set

	def OutputData(self, fp_path):
		"""
		output train result to target file

		:param fp_path:
		 	path of the target file

		output data fmt:
			b
			w
		"""

		with open(fp_path, 'w') as fp:
			fp.write(str(self.b) + '\n')
			l_value = []
			for v in self.w:
				l_value.append(str(v))
			fp.write('\t'.join(l_value) + '\n')

	def OutputHistory(self, fp_path):
		"""
		output the calculate history

		:param fp_path:
			path of the target file

		fmt:
			"the iteration ord", "the error item", "value of w", "value of b"
		"""

		with open(fp_path, 'w') as fp:
			for i in self.train_history:
				l_value = []
				for v in i:
					l_value.append(str(v))
				fp.write('\t'.join(l_value) + '\n')

	def Distance(self, item):
		"""
		calculate the functional distance between 'item' and the dicision surface.

		:param item:
		 	(n, ...), an item which is to be calculated.

		:return:
			yi(w*xi+b).
		"""

		res = 0

		for i in range(len(item[1])):
			res += item[1][i] * self.w[i]

		return (res + self.b) * item[0]

	def Update(self, item):
		"""
		update parameters(w, b) using stochastic gradient descent by item.

		:param item:
			(n, ...), an item which is classified into wrong class.
		"""
		for i in range(len(item[1])):
			self.w[i] += self.step * item[1][i] * item[0]

		self.b += self.step * item[0]

	def Calc(self, train_max_count = 0):
		"""
		calculate the value that the hyperplane can classify the traning set correctly
		under the indicate count.

		:param train_max_count:
			train_max_count: the max count to calculate, default equals to the train num * 10000

		:return:
			True or	False
		"""

		train_seq, train_num = 0, len(self.train_set)

		flag, train_ord = True, 0
		if train_max_count == 0:
			train_max_count = train_num * 10000

		while train_seq < train_num:
			if self.Distance(self.train_set[train_seq]) <= 0:
				self.train_history.append([train_seq] + self.w + [self.b])
				self.Update(self.train_set[train_seq])

				train_ord, train_seq = train_ord + 1, -1
				if train_ord >= train_max_count:
					flag = False
					break
			train_seq += 1

		self.train_history.append([-1] + self.w + [self.b])
		return flag

	def Parse(self, fl_input, fl_output = None, fl_history = None, max_train = 10000):
		"""
		generate the train data of peceptron

		:param fl_input:
			file that saves the data
		:param fl_output:
			file to save the result
		:param fl_history:
			file to save the train history
		:param max_train:
			the max train count to prevent from infinite loop
		"""

		self.InputData(fl_input)
		self.Calc(max_train)
		if fl_history:
			self.OutputHistory(fl_history)
		if fl_output:
			self.OutputData(fl_output)

if __name__ == '__main__':

	optparser = OptionParser()

	optparser.add_option('-i',
						 dest='input',
						 help='input data file or directory')

	optparser.add_option('-y',
						 dest='history',
						 help='file to save trained history',
						 default=None)

	optparser.add_option('-o',
						 dest='output',
						 help='output train result to file')

	optparser.add_option('-s',
						 dest='step',
						 help='the step length',
						 default=0.1,
						 type='float')

	optparser.add_option('-m',
						 dest='max',
						 help='the max train count',
						 default=10000,
						 type='int')

	(options, args) = optparser.parse_args()
	per_model = Perceptron(options.step)
	per_model.Parse(options.input, options.output, options.history, options.max)


