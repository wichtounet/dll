pipeline {
    agent any

    environment {
       CXX = "g++-6.4.0"
       LD = "g++-6.4.0"
       ETL_MKL = 'true'
    }

    stages {
        stage ('git'){
            steps {
                checkout([
                    $class: 'GitSCM',
                    branches: scm.branches,
                    doGenerateSubmoduleConfigurations: false,
                    extensions: scm.extensions + [[$class: 'SubmoduleOption', disableSubmodules: false, recursiveSubmodules: true, reference: '', trackingSubmodules: false]],
                    submoduleCfg: [],
                    userRemoteConfigs: scm.userRemoteConfigs])
            }
        }

        stage ('pre-analysis') {
            steps {
                sh 'cppcheck --xml-version=2 --enable=all --std=c++11 include/dll/*.hpp test/src/*.cpp test_compile/*.cpp 2> cppcheck_report.xml'
                sh 'sloccount --duplicates --wide --details include/dll/*.hpp test/src/*.cpp test_compile/*.cpp  > sloccount.sc'
                sh 'cccc include/dll/*.hpp test/*.cpp test_compile/*.cpp || true'
            }
        }

        stage ('build'){
            steps {
                sh 'make clean'
                sh 'CXX=g++-6.3.0 LD=g++-6.3.0 ETL_MKL=true make -j6 release_debug'
            }
        }

        stage ('test'){
            steps {
                sh "LD_LIBRARY_PATH=\"${env.LD_LIBRARY_PATH}:/opt/intel/mkl/lib/intel64:/opt/intel/lib/intel64\" ./release_debug/bin/dll_test_unit -r junit -d yes -o catch_report.xml || true"
                archive 'catch_report.xml'
                junit 'catch_report.xml'
            }
        }

        stage ('sonar-master'){
            when {
                branch 'master'
            }
            steps {
                sh "/opt/sonar-runner/bin/sonar-runner"
            }
        }

        stage ('sonar-branch'){
            when {
                not {
                    branch 'master'
                }
            }
            steps {
                sh "/opt/sonar-runner/bin/sonar-runner -Dsonar.branch=${env.BRANCH_NAME}"
            }
        }
    }

    post {
        always {
            script {
                if (currentBuild.result == null) {
                    currentBuild.result = 'SUCCESS'
                }
            }

            step([$class: 'Mailer',
                notifyEveryUnstableBuild: true,
                recipients: "baptiste.wicht@gmail.com",
                sendToIndividuals: true])
        }
    }
}
