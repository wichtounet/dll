node {
   try {
       stage 'git'
       checkout([$class: 'GitSCM', branches: scm.branches, doGenerateSubmoduleConfigurations: false, extensions: scm.extensions + [[$class: 'SubmoduleOption', disableSubmodules: false, recursiveSubmodules: true, reference: '', trackingSubmodules: false]], submoduleCfg: [], userRemoteConfigs: scm.userRemoteConfigs])

       stage 'pre-analysis'
       sh 'cppcheck --xml-version=2 --enable=all --std=c++11 include/dll/*.hpp test/src/*.cpp test_compile/*.cpp 2> cppcheck_report.xml'
       sh 'sloccount --duplicates --wide --details include/dll/*.hpp test/src/*.cpp test_compile/*.cpp  > sloccount.sc'
       sh 'cccc include/dll/*.hpp test/*.cpp test_compile/*.cpp || true'

       env.CXX="g++-4.9.4"
       env.LD="g++-4.9.4"
       env.ETL_MKL='true'
       env.DLL_COVERAGE='true'
       env.LD_LIBRARY_PATH="${env.LD_LIBRARY_PATH}:/opt/intel/mkl/lib/intel64"
       env.LD_LIBRARY_PATH="${env.LD_LIBRARY_PATH}:/opt/intel/lib/intel64"

       stage 'build'
       sh 'make clean'
       sh 'make -j6 release_debug'

       stage 'test'
       sh './release_debug/bin/dll_test_unit -r junit -d yes -o catch_report.xml || true'
       sh 'gcovr -x -b -r . --object-directory=release_debug/test > coverage_report.xml'
       archive 'catch_report.xml'


       stage 'sonar'
       if (env.BRANCH_NAME == "master") {
           sh '/opt/sonar-runner/bin/sonar-runner'
       } else {
           sh '/opt/sonar-runner/bin/sonar-runner' -Dsonar.branch=env.BRANCH_NAME
       }

       currentBuild.result = 'SUCCESS'
   } catch (any) {
       currentBuild.result = 'FAILURE'
       throw any
   } finally {
       step([$class: 'Mailer',
           notifyEveryUnstableBuild: true,
           recipients: "baptiste.wicht@gmail.com",
           sendToIndividuals: true])
   }
}
