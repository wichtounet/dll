debug_test_crbm_mp/mnist_1 : debug_dll_test
	 @ echo "Run crbm_mp/mnist_1" > test_reports/test_crbm_mp-mnist_1.log
	 @ ./debug/bin/dll_test crbm_mp/mnist_1 >> test_reports/test_crbm_mp-mnist_1.log

release_test_crbm_mp/mnist_1 : release_dll_test
	 @ echo "Run crbm_mp/mnist_1" > test_reports/test_crbm_mp-mnist_1.log
	 @ ./release/bin/dll_test crbm_mp/mnist_1 >> test_reports/test_crbm_mp-mnist_1.log

debug_test_crbm/mnist_1 : debug_dll_test
	 @ echo "Run crbm/mnist_1" > test_reports/test_crbm-mnist_1.log
	 @ ./debug/bin/dll_test crbm/mnist_1 >> test_reports/test_crbm-mnist_1.log

release_test_crbm/mnist_1 : release_dll_test
	 @ echo "Run crbm/mnist_1" > test_reports/test_crbm-mnist_1.log
	 @ ./release/bin/dll_test crbm/mnist_1 >> test_reports/test_crbm-mnist_1.log

debug_test_all: debug_test_crbm_mp/mnist_1 debug_test_crbm/mnist_1 
	 @ bash ./tools/test_report.sh

release_test_all: release_test_crbm_mp/mnist_1 release_test_crbm/mnist_1 
	 @ bash ./tools/test_report.sh

.PHONY: release_test_all debug_test_alldebug_test_crbm_mp/mnist_1 release_test_crbm_mp/mnist_1 debug_test_crbm/mnist_1 release_test_crbm/mnist_1 
