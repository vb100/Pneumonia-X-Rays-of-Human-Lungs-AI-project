# pneumonia_xrays
This projects includes both back-end (Keras, Flask with dependencies) and front-end/deployement (HTML, CSS, JS) parts. The project perform X-rays of human lungs classifiaction using Convolutional Neural Networks with automated image pre-processing and internal procedures. Saved files of inspected models are saved on specific folders in project directory.

The accuracy of the model is approximately 80% (with the help of some regularization techniques as Dropout, L2 Regularization and Rmsprop with Adam). 

For deploying I decided to use Flask instead of Django because of it simplicity and compatibility with Amazon Web Services (AWS) and Google Cloud Storage (GCS). That works fine. App Engine in GCS is a perfect tool for deploying a Keras model as web application. Do not forget to carefully upload your deployment environment and crate Python YAML file that the server understand what is configuration of your application.

Important part is to understand how to upload an image file to the server (Google bucket) with Flask. By this, I modified permission for app. bucket to let user write and read (when app makes a prediction) files there.

Some additional work with HTML, CSS and Javascript and the model frontend looks bright, simply and user friendly, even compatible with mobile phones.

Link to web application: https://flaskdeploy-xray.appspot.com/
Youtube link: https://youtu.be/XolgXocLOd8
LinkedIn profile: https://www.linkedin.com/in/bielinskas/
