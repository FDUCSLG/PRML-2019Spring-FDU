import os
os.sys.path.append('..')
from handout import get_data
from scipy.optimize import minimize

from source import *
from assessment import *

NUM_TEST_BASE = 10000

# question 1
def sample_data_influence():
	num_data = 100 
	sample_data1 = get_data(100)
	sample_data2 = get_data(500)
	sample_data3 = get_data(1000)
	sample_data4 = get_data(10000)

	title = "histogram: bins=50, num_sd=100"
	histogram_estimation(num_bins=50, sample_data=sample_data1, status=True, title=title)
	title = "histogram: bins=50, num_sd=500"
	histogram_estimation(num_bins=50, sample_data=sample_data2, status=True, title=title)
	title = "histogram: bins=50, num_sd=1000"
	histogram_estimation(num_bins=50, sample_data=sample_data3, status=True, title=title)
	title = "histogram: bins=50, num_sd=10000"
	histogram_estimation(num_bins=50, sample_data=sample_data4, status=True, title=title)


	title = "KDE: h=0.4, num_sd=100"
	kernel_density_estimation(h=0.4, sample_data=sample_data1, num_test_base=NUM_TEST_BASE, status=True, title=title)
	title = "KDE: h=0.4, num_sd=500"
	kernel_density_estimation(h=0.4, sample_data=sample_data2, num_test_base=NUM_TEST_BASE, status=True, title=title)
	title = "KDE: h=0.4, num_sd=1000"
	kernel_density_estimation(h=0.4, sample_data=sample_data3, num_test_base=NUM_TEST_BASE, status=True, title=title)
	title = "KDE: h=0.4, num_sd=10000"
	kernel_density_estimation(h=0.4, sample_data=sample_data4, num_test_base=NUM_TEST_BASE, status=True, title=title)


	title = "K=20, num_sd=100"
	knn_estimation(K=20, sample_data=sample_data1, num_test_base=NUM_TEST_BASE, status=True, title=title)
	title = "K=20, num_sd=500"
	knn_estimation(K=20, sample_data=sample_data2, num_test_base=NUM_TEST_BASE, status=True, title=title)
	title = "K=20, num_sd=1000"
	knn_estimation(K=20, sample_data=sample_data3, num_test_base=NUM_TEST_BASE, status=True, title=title)
	title = "K=20, num_sd=10000"
	knn_estimation(K=20, sample_data=sample_data4, num_test_base=NUM_TEST_BASE, status=True, title=title)


# question2
def histogram_exploration():
	num_data = 200 
	sample_data = get_data(num_data)

	title = "bins=10, num_sd=200"
	histogram_estimation(num_bins=10, sample_data=sample_data, status=False, title=title)

	title = "bins=25, num_sd=200"
	histogram_estimation(num_bins=25, sample_data=sample_data, status=False, title=title)

	title = "bins=100, num_sd=200"
	histogram_estimation(num_bins=100, sample_data=sample_data, status=False, title=title)

def histogram_bins_selection():
	num_data = 200 
	sample_data = get_data(num_data)

	num_bins = int(square_root_choice(sample_data))
	title = "square_root_choice: " + "bins=" + str(num_bins) + ", num_sd=200"
	histogram_estimation(num_bins=num_bins, sample_data=sample_data, status=True, title=title)

	num_bins = int(sturges_formula(sample_data))
	title = "sturges_formula: " + "bins=" + str(num_bins) + ", num_sd=200"
	histogram_estimation(num_bins=num_bins, sample_data=sample_data, status=True, title=title)

	num_bins = int(rice_rule(sample_data))
	title = "rice_rule: " + "bins=" + str(num_bins) + ", num_sd=200"
	histogram_estimation(num_bins=num_bins, sample_data=sample_data, status=True, title=title)

	num_bins = int(scotts_normal_reference_rule(sample_data))
	title = "scotts_normal_reference_rule: " + "bins=" + str(num_bins) + ", num_sd=200"
	histogram_estimation(num_bins=num_bins, sample_data=sample_data, status=True, title=title)

	num_bins = int(shimazaki_and_shinomoto(sample_data))
	title = "shimazaki_and_shinomoto: " + "bins=" + str(num_bins) + ", num_sd=200"
	plt.cla()
	histogram_estimation(num_bins=num_bins, sample_data=sample_data, status=True, title=title)
# question 3
def KDE_exploration():
	num_data = 200 
	sample_data = get_data(num_data)

	title = "h=0.1, num_sd=200"
	kernel_density_estimation(h=0.1, sample_data=sample_data, num_test_base=NUM_TEST_BASE, status=False, title=title)

	title = "h=0.4, num_sd=200"
	kernel_density_estimation(h=0.4, sample_data=sample_data, num_test_base=NUM_TEST_BASE, status=False, title=title)

	title = "h=1.0, num_sd=200"
	kernel_density_estimation(h=1.0, sample_data=sample_data, num_test_base=NUM_TEST_BASE, status=False, title=title)

def KDE_h_selection():
	num_data = 200
	sample_data = get_data(num_data)

	# mathematic assumption
	h = hMISE(sample_data)
	title = "AMISE: " + "h=" + str(h) + ", num_sd=200"
	kernel_density_estimation(h=h, sample_data=sample_data, num_test_base=NUM_TEST_BASE, status=True, title=title)

	# MLCV
	h = 0.2
	sample_data = get_data(100)
	sol = minimize(MLCV_KDE, h, sample_data)
	h = sol['x'][0]
	title = "MLCV: " + "h=" + str(h) + ", num_sd=100"
	kernel_density_estimation(h=h, sample_data=sample_data, num_test_base=NUM_TEST_BASE, status=True, title=title)
	
	h = 0.2
	sample_data = get_data(200)
	sol = minimize(MLCV_KDE, h, sample_data)
	h = sol['x'][0]
	title = "MLCVL: " + "h=" + str(h) + ", num_sd=200"
	kernel_density_estimation(h=h, sample_data=sample_data, num_test_base=NUM_TEST_BASE, status=True, title=title)

	h = 0.2
	sample_data = get_data(1000)
	sol = minimize(MLCV_KDE, h, sample_data)
	h = sol['x'][0]
	title = "MLCV: " + "h=" + str(h) + ", num_sd=500"
	kernel_density_estimation(h=h, sample_data=sample_data, num_test_base=NUM_TEST_BASE, status=True, title=title)


# question 4
def knn_exploration():
	num_data = 200
	sample_data = get_data(num_data)

	title = "K=1, num_sd=200"
	knn_estimation(K=1, sample_data=sample_data, num_test_base=NUM_TEST_BASE, status=True, title=title)

	title = "K=5, num_sd=200"
	knn_estimation(K=5, sample_data=sample_data, num_test_base=NUM_TEST_BASE, status=True, title=title)

	title = "K=15, num_sd=200"
	knn_estimation(K=15, sample_data=sample_data, num_test_base=NUM_TEST_BASE, status=True, title=title)

	title = "K=20, num_sd=200"
	knn_estimation(K=20, sample_data=sample_data, num_test_base=NUM_TEST_BASE, status=True, title=title)

	title = "K=100, num_sd=200"
	knn_estimation(K=100, sample_data=sample_data, num_test_base=NUM_TEST_BASE, status=True, title=title)


# sample_data_influence()

histogram_exploration()
histogram_bins_selection()
KDE_exploration()
KDE_h_selection()
knn_exploration()


