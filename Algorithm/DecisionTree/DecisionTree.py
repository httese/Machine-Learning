#!/usr/bin/python
#encoding=utf-8

"""
A Python implementation of the decision tree algorithm.
"""

from copy import *
from math import *
import numpy as np
import pandas as pd
from optparse import OptionParser


def StatisCount(lO):

	dObj = {}

	for obj in lO:
		if obj not in dObj:
			dObj[obj] = 1
		else:
			dObj[obj] += 1

	return dObj


def StatisDistribution(lO):
	"""
	Statis the obj list, and return the relative probability list

	:param lO: the obj list
	:return: the relative probability list
	"""

	cnt, lp, dObj = float(len(lO)), [], StatisCount(lO)

	for obj in dObj:
		lp.append(dObj[obj] / cnt)

	return lp


def EntropyX(lp, base=e):
	"""
	H(p)

	:param lp:
	:param base:
	:return:
	"""

	hp = 0.

	for p in lp:
		if p == 0:
			continue
		hp -= p * log(p, base)

	return hp


def EntropyXY(llp, lw, base=e):
	"""
	H(D|A)

	g(D,A) = H(D) - H(D|A)

	:param llp:
	:param lw:
	:param base:
	:return:
	"""

	hp = 0.

	for lp, w in zip(llp, lw):
		hp += EntropyX(lp, base) * w

	return hp


def GiniX(lp):
	return 1 - np.power(lp, 2).sum()


class ID3Node:

	def __init__(self, type, entropy, cnt, value):
		self.type = type
		self.entropy = entropy
		self.cnt = cnt
		self.dSub = value

		self.hight = 0
		self.parent = None
		self.dClassify = {}

		if len(value.keys()) == 0:
			self.dClassify[type] = cnt


class ID3:
	"""
	C4.5 and CART derive from ID3.

	"""

	def __init__(self, base=e, threshold=0.0, alpha=0.0):
		self.df = None
		self.tree = None

		self.base = base
		self.threshold = float(threshold)
		self.alpha = float(alpha)

	def __del__(self):
		pass

	def InputData(self, fl_path):
		self.df = pd.read_table(fl_path, index_col=0)

	def Generate(self):

		lObj = [(self.tree, None, deepcopy(self.df))]
		while len(lObj):
			(ndParent, valParent, df) = lObj.pop(0)
			nd = self.NodeCalc(df)

			# collerate
			if ndParent == None:
				self.tree = nd
			else:
				ndParent.dSub[valParent] = nd
				nd.parent = ndParent

			# sub node
			for v in nd.dSub:
				dfSub = deepcopy(df[df[nd.type] == v])
				lObj.append((nd, v, dfSub.drop(nd.type, axis=1)))

	def NodeCalc(self, df):

		# calculate the entropy of class
		entroryCls = EntropyX(StatisDistribution(df.index), self.base)

		# check
		if entroryCls == 0. or len(df.columns) == 0:
			lObj, dObj = [], StatisCount(df.index)
			for k in dObj:
				lObj.append((k,dObj[k]))
			lObj.sort(key= lambda i: i[1])
			return ID3Node(lObj[-1][0], 0, len(df.values), {})

		# calculate the entropy of obj
		dC, entroryObj = StatisCount(df.index), []
		for col in df.columns:
			llp, lw, dObj = [], [], StatisCount(df[col])
			for obj in dObj:
				lp = []
				lw.append(float(dObj[obj]) / len(df.index))
				for c in dC:
					cnt = len(df[ (df[col] == obj) & (df.index == c) ])
					lp.append(float(cnt) / dObj[obj])
				llp.append(lp)
			entroryObj.append((col, EntropyXY(llp, lw, self.base)))

		# choice the min
		entroryObj.sort(key= lambda i: i[1])
		info = entroryObj[0]
		if info[1] < self.threshold:
			lObj, dObj = [], StatisCount(df.index)
			lObj.append((k,dObj[k]) for k in dObj)
			lObj.sort(key= lambda i: i[1])
			return ID3Node(lObj[-1][0], info[1], len(df.values), {})

		# create new node
		return ID3Node(info[0], info[1], len(df.values), StatisCount(df[info[0]]))

	def OutputData(self, fl):
		lNode = [('', '', self.tree)]

		with open(fl, 'w') as fp:
			while len(lNode):
				(type, val, nd) = lNode.pop(0)
				fp.write('%s[%s %d]: %s\n' % (type, val, nd.cnt, nd.type))

				for k in nd.dSub:
					lNode.append((nd.type, k, nd.dSub[k]))

	def Pruning(self):

		bCon = True
		while bCon:
			# list leaf node
			lLeafNode, lNode = [], [self.tree]
			while len(lNode):
				nd = lNode.pop(0)
				for k in nd.dSub:
					lNode.append(nd.dSub[k])

				if len(nd.dSub.keys()) == 0:
					lLeafNode.append(nd)

			lInternal = []
			for leaf in lLeafNode:
				if leaf.parent not in lInternal and leaf.parent != None:
					lInternal.append(leaf.parent)

			if self.tree in lInternal:
				lInternal.remove(self.tree)
			if len(lInternal) < 1:
				break

			lInternal.sort(key= lambda i: i.entropy)
			# calculate the loss(uncertain) value
			for nd in lInternal:
				(aT, at) = self.NodeValueOfPurning(nd)
				if np.multiply(np.subtract(aT, at), [1, self.alpha]).sum() < 0: # aT < at
					continue

				cnt, type = 0, None
				for t in nd.dClassify:
					if nd.dClassify[t] > cnt:
						cnt, type = nd.dClassify[t], t

				nd.dSub, nd.dClassify = {}, {}
				nd.dClassify[type] = nd.cnt
				nd.entropy = 0.0
				nd.type = type

				bCon = True

	def NodeValueOfPurning(self, node):

		lLeafNode, lNode = [], [node]
		while len(lNode):
			nd = lNode.pop(0)
			for k in nd.dSub:
				lNode.append(nd.dSub[k])

			if len(nd.dSub.keys()) == 0:
				lLeafNode.append(nd)

				if nd.type not in node.dClassify:
					node.dClassify[nd.type] = nd.cnt
				else:
					node.dClassify[nd.type] += nd.cnt

		# get the loss value before pruning the node
		aT = sum([self.NodeLossValue(nd) for nd in lLeafNode])

		# get the loss value after prsuning the node
		at = self.NodeLossValue(node)

		return ((aT, len(lLeafNode)), (at, 1))

	def NodeLossValue(self, node):
		lp = []
		for val in node.dClassify:
			lp.append(node.dClassify[val] / float(node.cnt))
		return node.cnt * EntropyX(lp, self.base)


	def Query(self):
		pass


class ICARTNode:

	def __init__(self, type, cnt, value, gini):
		self.type = type
		self.cnt = cnt
		self.val = value
		self.gini = gini

		self.parent = None
		self.dSub = {}


	def __del__(self):
		pass


class ICART:

	def __init__(self, base=e, threshold=0.0, alpha=0.0):
		self.df = None
		self.tree = None

		self.base = base
		self.threshold = float(threshold)
		self.alpha = float(alpha)


	def __del__(self):
		pass


	def InputData(self, fl_path):
		self.df = pd.read_table(fl_path, index_col=0)


	def Generate(self):
		print(len(self.df.columns))
		df = self.Filter(deepcopy(self.df))
		print(len(df.columns))

		lObj = [(self.tree, None, df)]
		while len(lObj):
			(ndParent, valParent, df) = lObj.pop(0)
			nd = self.NodeCalc(df)

			# collerate
			if ndParent == None:
				self.tree = nd
			else:
				ndParent.dSub[valParent] = nd
				nd.parent = ndParent

			dfG, dfLE = df[df[nd.type] > nd.val], df[df[nd.type] <= nd.val]
			if len(dfG) and len(set(dfG.index)) > 1:
				lObj.append((nd, 'G', deepcopy(dfG)))
			else:
				nd.dSub['G'] = ICARTNode('', nd.cnt - len(dfLE), None, None)

			if len(dfLE) and len(set(dfLE.index)) > 1:
				lObj.append((nd, 'LE', deepcopy(dfLE)))
			else:
				nd.dSub['LE'] = ICARTNode('', nd.cnt - len(dfG), None, None)

	def Filter(self, df):
		lCnt = []
		for col in df.columns:
			lCnt.append((col, set(df[col])))

		for obj in lCnt:
			if len(obj[1]) == 1:
				df = df.drop(obj[0], axis=1)

		return df


	def NodeCalc(self, df):

		# calculate the gini of class
		minCol, cnt, minObj, minGini = None, float(len(df)), None, GiniX(StatisDistribution(df.index))

		# calculate the gini of obj
		for col in df.columns:
			lObj = list(set(df[col]))
			lObj.sort()
			#优化lObj,选择最小值和最大值

			for obj in lObj:
				dfG, dfLE = df[df[col] > obj], df[df[col] <= obj]
				giniObj = (len(dfG) / cnt) * GiniX(StatisDistribution(dfG.index)) + \
						  (len(dfLE) / cnt) * GiniX(StatisDistribution(dfLE.index))

				if giniObj < minGini:
					minCol, minObj, minGini = col, obj, giniObj

		return ICARTNode(minCol, cnt, minObj, minGini)


	def OutputData(self, fl):
		lNode = [('', '', '', self.tree)]

		with open(fl, 'w') as fp:
			while len(lNode):
				(op, type, val, nd) = lNode.pop(0)
				fp.write('%s %s %s = %s[%s]\n' % (op, type, val, nd.cnt, nd.type))

				for k in nd.dSub:
					lNode.append((k, nd.type, nd.val, nd.dSub[k]))


if __name__ == '__main__':

	optparser = OptionParser()

	optparser.add_option('-i',
						 dest='input',
						 help='input data file or directory')

	optparser.add_option('-b',
						 dest='base',
						 default=2.0,
						 type=float,
						 help='the base value')

	optparser.add_option('-a',
						 dest='alpha',
						 default=2.0,
						 type=float,
						 help='the alpha value')

	optparser.add_option('-t',
						 dest='threshold',
						 default=0.0,
						 type=float,
						 help='the threshold value')

	(options, args) = optparser.parse_args()

	fl, tmID3 = options.input, ID3(options.base, threshold=options.threshold, alpha=options.alpha)

	tmID3.InputData(fl)

	tmID3.Generate()
	tmID3.OutputData(fl + '.tree')

	tmID3.Pruning()
	tmID3.OutputData(fl + '.pruning')
	
	
	tmCART = ICART(2)
	tmCART.InputData(fl)
	tmCART.Generate()
	tmCART.OutputData(fl + '.tree')

