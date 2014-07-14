debug_test_rbm/mnist_1 : debug_dll_test
	 @ echo "Run rbm/mnist_1" > test_reports/test_rbm-mnist_1.log
	 @ ./debug/bin/dll_test rbm/mnist_1 >> test_reports/test_rbm-mnist_1.log

release_test_rbm/mnist_1 : release_dll_test
	 @ echo "Run rbm/mnist_1" > test_reports/test_rbm-mnist_1.log
	 @ ./release/bin/dll_test rbm/mnist_1 >> test_reports/test_rbm-mnist_1.log

debug_test_rbm/mnist_2 : debug_dll_test
	 @ echo "Run rbm/mnist_2" > test_reports/test_rbm-mnist_2.log
	 @ ./debug/bin/dll_test rbm/mnist_2 >> test_reports/test_rbm-mnist_2.log

release_test_rbm/mnist_2 : release_dll_test
	 @ echo "Run rbm/mnist_2" > test_reports/test_rbm-mnist_2.log
	 @ ./release/bin/dll_test rbm/mnist_2 >> test_reports/test_rbm-mnist_2.log

debug_test_rbm/mnist_3 : debug_dll_test
	 @ echo "Run rbm/mnist_3" > test_reports/test_rbm-mnist_3.log
	 @ ./debug/bin/dll_test rbm/mnist_3 >> test_reports/test_rbm-mnist_3.log

release_test_rbm/mnist_3 : release_dll_test
	 @ echo "Run rbm/mnist_3" > test_reports/test_rbm-mnist_3.log
	 @ ./release/bin/dll_test rbm/mnist_3 >> test_reports/test_rbm-mnist_3.log

debug_test_crbm_mp/mnist_1 : debug_dll_test
	 @ echo "Run crbm_mp/mnist_1" > test_reports/test_crbm_mp-mnist_1.log
	 @ ./debug/bin/dll_test crbm_mp/mnist_1 >> test_reports/test_crbm_mp-mnist_1.log

release_test_crbm_mp/mnist_1 : release_dll_test
	 @ echo "Run crbm_mp/mnist_1" > test_reports/test_crbm_mp-mnist_1.log
	 @ ./release/bin/dll_test crbm_mp/mnist_1 >> test_reports/test_crbm_mp-mnist_1.log

debug_test_crbm_mp/mnist_2 : debug_dll_test
	 @ echo "Run crbm_mp/mnist_2" > test_reports/test_crbm_mp-mnist_2.log
	 @ ./debug/bin/dll_test crbm_mp/mnist_2 >> test_reports/test_crbm_mp-mnist_2.log

release_test_crbm_mp/mnist_2 : release_dll_test
	 @ echo "Run crbm_mp/mnist_2" > test_reports/test_crbm_mp-mnist_2.log
	 @ ./release/bin/dll_test crbm_mp/mnist_2 >> test_reports/test_crbm_mp-mnist_2.log

debug_test_crbm_mp/mnist_3 : debug_dll_test
	 @ echo "Run crbm_mp/mnist_3" > test_reports/test_crbm_mp-mnist_3.log
	 @ ./debug/bin/dll_test crbm_mp/mnist_3 >> test_reports/test_crbm_mp-mnist_3.log

release_test_crbm_mp/mnist_3 : release_dll_test
	 @ echo "Run crbm_mp/mnist_3" > test_reports/test_crbm_mp-mnist_3.log
	 @ ./release/bin/dll_test crbm_mp/mnist_3 >> test_reports/test_crbm_mp-mnist_3.log

debug_test_crbm_mp/mnist_4 : debug_dll_test
	 @ echo "Run crbm_mp/mnist_4" > test_reports/test_crbm_mp-mnist_4.log
	 @ ./debug/bin/dll_test crbm_mp/mnist_4 >> test_reports/test_crbm_mp-mnist_4.log

release_test_crbm_mp/mnist_4 : release_dll_test
	 @ echo "Run crbm_mp/mnist_4" > test_reports/test_crbm_mp-mnist_4.log
	 @ ./release/bin/dll_test crbm_mp/mnist_4 >> test_reports/test_crbm_mp-mnist_4.log

debug_test_crbm/mnist_1 : debug_dll_test
	 @ echo "Run crbm/mnist_1" > test_reports/test_crbm-mnist_1.log
	 @ ./debug/bin/dll_test crbm/mnist_1 >> test_reports/test_crbm-mnist_1.log

release_test_crbm/mnist_1 : release_dll_test
	 @ echo "Run crbm/mnist_1" > test_reports/test_crbm-mnist_1.log
	 @ ./release/bin/dll_test crbm/mnist_1 >> test_reports/test_crbm-mnist_1.log

debug_test_crbm/mnist_2 : debug_dll_test
	 @ echo "Run crbm/mnist_2" > test_reports/test_crbm-mnist_2.log
	 @ ./debug/bin/dll_test crbm/mnist_2 >> test_reports/test_crbm-mnist_2.log

release_test_crbm/mnist_2 : release_dll_test
	 @ echo "Run crbm/mnist_2" > test_reports/test_crbm-mnist_2.log
	 @ ./release/bin/dll_test crbm/mnist_2 >> test_reports/test_crbm-mnist_2.log

debug_test_crbm/mnist_3 : debug_dll_test
	 @ echo "Run crbm/mnist_3" > test_reports/test_crbm-mnist_3.log
	 @ ./debug/bin/dll_test crbm/mnist_3 >> test_reports/test_crbm-mnist_3.log

release_test_crbm/mnist_3 : release_dll_test
	 @ echo "Run crbm/mnist_3" > test_reports/test_crbm-mnist_3.log
	 @ ./release/bin/dll_test crbm/mnist_3 >> test_reports/test_crbm-mnist_3.log

debug_test_crbm/mnist_4 : debug_dll_test
	 @ echo "Run crbm/mnist_4" > test_reports/test_crbm-mnist_4.log
	 @ ./debug/bin/dll_test crbm/mnist_4 >> test_reports/test_crbm-mnist_4.log

release_test_crbm/mnist_4 : release_dll_test
	 @ echo "Run crbm/mnist_4" > test_reports/test_crbm-mnist_4.log
	 @ ./release/bin/dll_test crbm/mnist_4 >> test_reports/test_crbm-mnist_4.log

debug_test_all: debug_test_rbm/mnist_1 debug_test_rbm/mnist_2 debug_test_rbm/mnist_3 debug_test_crbm_mp/mnist_1 debug_test_crbm_mp/mnist_2 debug_test_crbm_mp/mnist_3 debug_test_crbm_mp/mnist_4 debug_test_crbm/mnist_1 debug_test_crbm/mnist_2 debug_test_crbm/mnist_3 debug_test_crbm/mnist_4 
	 @ bash ./tools/test_report.sh

release_test_all: release_test_rbm/mnist_1 release_test_rbm/mnist_2 release_test_rbm/mnist_3 release_test_crbm_mp/mnist_1 release_test_crbm_mp/mnist_2 release_test_crbm_mp/mnist_3 release_test_crbm_mp/mnist_4 release_test_crbm/mnist_1 release_test_crbm/mnist_2 release_test_crbm/mnist_3 release_test_crbm/mnist_4 
	 @ bash ./tools/test_report.sh

.PHONY: release_test_all debug_test_alldebug_test_rbm/mnist_1 release_test_rbm/mnist_1 debug_test_rbm/mnist_2 release_test_rbm/mnist_2 debug_test_rbm/mnist_3 release_test_rbm/mnist_3 debug_test_crbm_mp/mnist_1 release_test_crbm_mp/mnist_1 debug_test_crbm_mp/mnist_2 release_test_crbm_mp/mnist_2 debug_test_crbm_mp/mnist_3 release_test_crbm_mp/mnist_3 debug_test_crbm_mp/mnist_4 release_test_crbm_mp/mnist_4 debug_test_crbm/mnist_1 release_test_crbm/mnist_1 debug_test_crbm/mnist_2 release_test_crbm/mnist_2 debug_test_crbm/mnist_3 release_test_crbm/mnist_3 debug_test_crbm/mnist_4 release_test_crbm/mnist_4 
