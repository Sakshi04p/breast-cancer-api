// Jenkinsfile
// Jenkins Pipeline for Breast Cancer Prediction FastAPI App

pipeline {
    agent any

    stages {

        stage('Clone Repository') {
            steps {
                echo 'Cloning project...'
                // Uncomment below line if using Git:
                // git 'https://github.com/your-username/breast-cancer-api.git'
                git branch: 'main', url: 'https://github.com/Sakshi04p/breast-cancer-api.git'
                echo 'Project files ready.'
            }
        }

        stage('Install Dependencies') {
            steps {
                echo 'Installing Python dependencies...'
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Train Model') {
            steps {
                echo 'Training ML model and saving pickle file...'
                sh 'python train_model.py'
            }
        }

        stage('Run FastAPI App') {
            steps {
                echo 'Starting FastAPI application on port 8000...'
                sh 'nohup uvicorn main:app --host 0.0.0.0 --port 8000 &'
                sleep(time: 3, unit: 'SECONDS')
                echo 'App running at http://localhost:8000'
            }
        }

        stage('Test Application') {
            steps {
                echo 'Testing /predict endpoint with sample input...'
                sh '''
                curl -X POST "http://localhost:8000/predict" \
                     -H "Content-Type: application/json" \
                     -d '{
                           "mean_radius": 17.99, "mean_texture": 10.38,
                           "mean_perimeter": 122.8, "mean_area": 1001.0,
                           "mean_smoothness": 0.1184, "mean_compactness": 0.2776,
                           "mean_concavity": 0.3001, "mean_concave_points": 0.1471,
                           "mean_symmetry": 0.2419, "mean_fractal_dimension": 0.07871,
                           "radius_se": 1.095, "texture_se": 0.9053,
                           "perimeter_se": 8.589, "area_se": 153.4,
                           "smoothness_se": 0.006399, "compactness_se": 0.04904,
                           "concavity_se": 0.05373, "concave_points_se": 0.01587,
                           "symmetry_se": 0.03003, "fractal_dimension_se": 0.006193,
                           "worst_radius": 25.38, "worst_texture": 17.33,
                           "worst_perimeter": 184.6, "worst_area": 2019.0,
                           "worst_smoothness": 0.1622, "worst_compactness": 0.6656,
                           "worst_concavity": 0.7119, "worst_concave_points": 0.2654,
                           "worst_symmetry": 0.4601, "worst_fractal_dimension": 0.1189
                         }'
                '''
            }
        }
    }

    post {
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed. Check the logs above.'
        }
    }
}
