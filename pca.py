import numpy as np
import pylab as pl
import xlrd
import sys
from math import sqrt
import pylab as pl
import pylab as plt
import copy
from matplotlib.pyplot import *
import matplotlib.patches as mpatches
from matplotlib.mlab import PCA as mlabPCA
from sklearn.decomposition import PCA as sklearnPCA
import matplotlib.lines as mlines


def populateStat(sheet, name, column_number):
    arr = []
    for i in range(sheet.nrows):
    	if(sheet.cell(i,column_number).value != '') and (sheet.cell(i,column_number).value != name):
		    arr.append(sheet.cell(i,column_number).value)
    return arr
	
def standardize(input):
	mean_input = mean(input)
	variance = var(input, mean_input)
	sdev = std(variance)
	
	for i in range(len(input)):
		input[i] = (input[i] - mean_input)/float(sdev)
	return input
def mean(input):
	"""
        Return the mean of the input data
		
        @type  input: list of numbers
        @param input: the data to average
        
        @rtype:   number
        @return:  the mean of the input data
	"""
		
	return sum(input) / float(len(input)) 
	
def var(input, mean):
	"""
        Return the variance of the input data
		
        @type  input: list of numbers
        @param input: the data to find the variance of
		@type  mean: number
        @param mean: mean of the input data set
        
        @rtype:   number
        @return:  the variance of the input data
	"""
	sum = 0.0
	for i in range(len(input)):
		sum = sum + ((input[i] - mean)**2)
	return sum/float(len(input))
	
def std(variance):
	"""
        Return the standard deviation of the input variance
		
        @type  variance: number
        @param variance: the variance of the data 
		
		@rtype:   number
        @return:  the standard deviation
	"""
	return sqrt(variance)

names = ['Win Percentage','Runs','Batting Walks', 'Stikeouts', 'Stolen Bases', 'Batting Average', 'On Base Percentage', 'Slugging Percentage', 'Innings Pitched', 'Hits', 'Pitching Walks', 'K/9', 'ERA', 'WHIP', 'Fielding Percentage' ]
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'b', 'g', 'r', 'c', 'm']
teams = ['Curry', 'Eastern Nazarene', 'Endicott' ,'Gordon' , 'Nichols' , 'Roger Williams' , 'Salve Regina', 'Wentworth', 'Western New England']
years = ['2017', '2016', '2015', '2014', '2013']
colors2 = ['purple', 'red', 'lightgreen', 'lightskyblue', 'darkgreen', 'darkblue','cyan','gold','magenta']
symbol = ['*', 'v', '^', '+', 'o']
	
	
#import excel worksheet
data = xlrd.open_workbook(sys.argv[1])

#assign sheets within workbooks to objects
winPercentagesSheet = data.sheet_by_index(0)
battingStatsSheet = data.sheet_by_index(1)
pitchingStatsSheet = data.sheet_by_index(2)
fieldingStatsSheet = data.sheet_by_index(3)

winPercentages = populateStat(winPercentagesSheet,'Percentage', 3)
#batting stats
runs = populateStat(battingStatsSheet,'Runs', 2)
batting_walks = populateStat(battingStatsSheet,'Walks', 3)
strikeouts = populateStat(battingStatsSheet,'Strikeouts', 4)
stolen_bases = populateStat(battingStatsSheet,'Stolen Bases', 5)
batting_average = populateStat(battingStatsSheet,'Batting Average', 6)
on_base_percentage = populateStat(battingStatsSheet,'On Base Percentage', 7)
slugging_percentage = populateStat(battingStatsSheet,'Slugging Percentage', 8)

#pitching stats
innings_pitched = populateStat(pitchingStatsSheet,'Innings Pitched', 2)
hits = populateStat(pitchingStatsSheet,'Hits', 3)
pitching_walks = populateStat(pitchingStatsSheet,'Walks',4)
strikeouts_through_nine = populateStat(pitchingStatsSheet,'Strikeouts Through 9', 5)
earned_run_average = populateStat(pitchingStatsSheet,'Earned Run Average', 6)
whip = populateStat(pitchingStatsSheet,'Walks Hits Per Inning Pitched', 7)

#felding staticmethod
fielding_percentage = populateStat(fieldingStatsSheet,'Fielding Percentage',2)

winPercentages = standardize(winPercentages)
runs = standardize(runs)
batting_walks = standardize(batting_walks)
strikeouts = standardize(strikeouts)
stolen_bases = standardize(stolen_bases)
batting_average = standardize(batting_average)
on_base_percentage = standardize(on_base_percentage)
slugging_percentage = standardize(slugging_percentage)
innings_pitched = standardize(innings_pitched)
hits = standardize(hits)
pitching_walks = standardize(pitching_walks)
strikeouts_through_nine = standardize(strikeouts_through_nine)
earned_run_average = standardize(earned_run_average)
whip = standardize(whip)
fielding_percentage = standardize(fielding_percentage)


#so the matrix is in not in scientific notation
np.set_printoptions(suppress=True, precision=3)

#combine all stats into one standardized matrix
standardized_matrix = np.vstack([winPercentages,runs,batting_walks,strikeouts,stolen_bases, batting_average, on_base_percentage, slugging_percentage,innings_pitched, hits, pitching_walks, strikeouts_through_nine, earned_run_average, whip, fielding_percentage])
assert standardized_matrix.shape == (15,45) , 'input matrix incorrect size'

sklearn_pca = sklearnPCA(n_components=2)
sklearn_transf = sklearn_pca.fit_transform(standardized_matrix.T)






#get the covariance matrix from the standardized matrix 
covalence_matrix = np.cov(standardized_matrix)
print "Covalence matrix"
print covalence_matrix
print "\n"
assert covalence_matrix.shape == (15,15) , 'covalence matrix incorrect size'

eigen_values, eig_vectors = np.linalg.eig(covalence_matrix)

sum_eig = sum(eigen_values)


for i in range(len(eigen_values)):
	print('Eigenvector {}: \n{}'.format(i+1, eig_vectors[i].reshape(15,1)))
	print('Eigenvalue {}: {}'.format(i+1, eigen_values[i]))
idx = eigen_values.argsort()[::-1]
eigenvalues = eigen_values[idx][:2]

eigenvectors = np.atleast_1d(eig_vectors[:,idx])[:,:2]

# Project the data onto principal components
X_transformed = eigenvectors.T.dot(standardized_matrix)




print '-------------'
print idx
print eigenvalues
print eigenvectors
print eigenvalues[0]/sum_eig
print eigenvalues[1]/sum_eig

print '------------'
print "method 2 output"
print X_transformed
print '\n'

print "sklearn output"
print sklearn_transf
print '\n'


yearIndex = 0
nameIndex = -1
for i in range(0,45):
	nameIndex = i % 9
	if(nameIndex == 0):
		yearIndex = yearIndex + 1
	pl.plot(X_transformed[0][i], X_transformed[1][i],symbol[yearIndex-1],color = colors2[nameIndex])	


for i in range(len(eigenvectors)):
	arrow = pl.arrow( 0,0, eigenvectors[i][0],eigenvectors[i][1], fc='k', ec=colors[i], head_width=0.001, head_length=0.01, facecolor = 'r', label=names[i] )
	pl.annotate(names[i], xy=( eigenvectors[i][0],eigenvectors[i][1]))


patches = []
yearPatches = []
for i in range(len(teams)):
	patches.append(mpatches.Patch(color=colors2[i], label=teams[i]))
for i in range(len(years)):
	patches.append(mlines.Line2D([], [],marker=symbol[i], markersize=15, label=years[i]))
		
pl.legend(handles=patches, bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
pl.axhline(y=0, color='k')
pl.axvline(x=0, color='k')
pl.show()
