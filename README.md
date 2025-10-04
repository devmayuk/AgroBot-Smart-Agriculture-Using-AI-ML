# AgroBot-Smart-Agriculture-Using-AI-ML
Agriculture is a major revenue-producing sector in India, but changes in weather and biological patterns can result in significant losses for farmers. 
Precision farming can help reduce these losses by using advanced technologies such as IoT, data mining, data analytics, and machine learning to collect data and
make predictions about weather conditions, soil types, and crop types. This paper focuses on the algorithms used to predict crop yield and crop cost, with the 
aim of achieving smart farming. The proposed work helps farmers to manage crops and harvest in a smart way, guiding them for efficient cultivation and achieving
high productivity at a low cost. The paper provides an integrated solution for farming, which helps farmers to pre-plan activities before cultivation and reduce
manual labor. This research emphasizes the relevance of precision farming and the application of sophisticated technology to boost agricultural output.

## Deployment Notes

- The Render deployment now uses Python 3.10 (see `runtime.txt`) because Python 3.7 is end-of-life and no longer supported by Render's default builders.
- The Gunicorn entrypoint now targets the `App.app` module directly so the worker starts correctly when Render runs the Procfile from the repository root.
