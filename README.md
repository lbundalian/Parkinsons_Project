# Parkinsons_Project
Biomarker based machine learning model for early diagnosis of Parkinson's diseases

To run the project, follow these steps:

1. docker build -t sp-parkinsons-api .
2. docker run -d -p 80:80 sp-parkinsons-api
3. then access the swagger/docs via localhost:80/docs or localhost/docs