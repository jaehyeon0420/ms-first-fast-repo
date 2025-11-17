pipeline {
    agent any
      
    environment {
        APP_NAME = "msai-firstpjt-fastapi-app"
        ACR_NAME = "backprojectacr"
        ACR_LOGIN_SERVER = "backprojectacr.azurecr.io"
    }

    stages {
        stage('Cleanup Workspace') {
            steps {
                cleanWs()
            }
        }
	
        stage('Checkout Code with LFS') {
    steps {
        script {
            // Git 체크아웃 먼저
            checkout([
                $class: 'GitSCM',
                branches: [[name: '*/master']],
                extensions: [
                    [$class: 'CloneOption', depth: 1, noTags: false, shallow: false]
                ],
                userRemoteConfigs: [[
                    url: 'https://github.com/jaehyeon0420/ms-first-fast-repo.git',
                    credentialsId: 'github-token'
                ]]
            ])
            
            // Git LFS 확인 및 파일 다운로드
            sh '''
                # Git LFS 설치되어 있는지 확인
                if command -v git-lfs &> /dev/null; then
                    echo "Git LFS found. Pulling LFS files..."
                    git lfs install
                    git lfs pull
                else
                    echo "Git LFS not found. Trying direct download..."
                    # GitHub Media URL로 직접 다운로드
                    curl -L -o app/model/maskrcnn_model_final.pth \
                        "https://media.githubusercontent.com/media/jaehyeon0420/ms-first-fast-repo/master/app/model/maskrcnn_model_final.pth"
                fi
                
                # 파일 크기 확인
                if [ -f "app/model/maskrcnn_model_final.pth" ]; then
                    FILE_SIZE=$(stat -c%s "app/model/maskrcnn_model_final.pth" 2>/dev/null || stat -f%z "app/model/maskrcnn_model_final.pth")
                    echo "Model file size: $FILE_SIZE bytes"
                    
                    if [ "$FILE_SIZE" -lt 1000 ]; then
                        echo "ERROR: Model file is too small (likely LFS pointer file)"
                        exit 1
                    fi
                else
                    echo "ERROR: Model file not found!"
                    exit 1
                fi
            '''
        }
    }
}

        stage('Build Docker Image') {
            steps {
                script {
                    def IMAGE_TAG = "${env.ACR_LOGIN_SERVER}/${env.APP_NAME}:${env.BUILD_NUMBER}"
                    sh "docker build -t ${IMAGE_TAG} ."
                }
            }
        }

        stage('Login to Azure & Push to ACR') {
            steps {
                withCredentials([
                    string(credentialsId: 'AZURE_APP_ID', variable: 'AZURE_APP_ID'),
                    string(credentialsId: 'AZURE_PASSWORD', variable: 'AZURE_PASSWORD'),
                    string(credentialsId: 'AZURE_TENANT_ID', variable: 'AZURE_TENANT_ID')
                ]) {
                    sh """
                    echo Azure login...
                    az login --service-principal -u $AZURE_APP_ID -p $AZURE_PASSWORD --tenant $AZURE_TENANT_ID

                    echo Logging into ACR...
                    az acr login --name ${env.ACR_NAME}

                    echo Pushing Docker image to ACR...
                    docker push ${env.ACR_LOGIN_SERVER}/${env.APP_NAME}:${env.BUILD_NUMBER}
                    """
                }
            }
        }
    }
}