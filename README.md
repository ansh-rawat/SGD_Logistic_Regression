# Implementation of Logistic Regression using Stochastic Gradient Descent from scratch without using scikit-learn.


Logistic Regression is a classification machine learning algorithm which uses squashing function to push the output values between 0 and 1. 
Using Stochastic Gradient Descent is really helpful when we have large amouts data as it does not require to load the whole data into the memory. And also it is faster
than traditional Gradient Descent.


To better understand how Logistic Regression work with SGD, I've implemented SGD Logistic Regression from scatch without using scikit-learn library.


Logistic Regression algorithm workflow:
1. Output values are calculated with thwe help of y=w.T * x + b (same as Linear Regression) using training data.
2. Sigmoid function is used to squash the values between 0 and 1.
3. Cost function (logloss fucntion) is used to determine the total Loss value.
4. Derivatives of w and b are calculated.
5. Weights and biases are updated.
6. The above steps are repeated until we get best fit values of w and b for our training data.
7. The values of w and b which are generated after above iterations are used to get predictions for our test data and validate perfomance of our model.


Some of the important mathematical equations used by logistic regression are mentioned below: 

Log loss function (a.k.a. Cost Function used in Logistic Regression)
𝑙𝑜𝑔𝑙𝑜𝑠𝑠=−1∗1/𝑛 Σ(𝑓𝑜𝑟 𝑒𝑎𝑐ℎ 𝑌𝑡,𝑌𝑝𝑟𝑒𝑑) (𝑌𝑡 * 𝑙𝑜𝑔10(𝑌𝑝𝑟𝑒𝑑) + (1−𝑌𝑡) * 𝑙𝑜𝑔10(1−𝑌𝑝𝑟𝑒𝑑)) 

Derivative of weights and biases 
𝑑𝑤 = 𝑥 * (𝑦 − σ(𝑤.𝑇 * 𝑥 + 𝑏) − λ/𝑁 * 𝑤) 
𝑑𝑏 = 𝑦 − σ((𝑤.𝑇 * 𝑥 + 𝑏)) 

Updatating weights and biases
𝑤(𝑡+1)←𝑤(𝑡)+α(𝑑𝑤(𝑡)) 
𝑏(𝑡+1)←𝑏(𝑡)+α(𝑑𝑏(𝑡)) 

To better understand above equations please go though this link: https://drive.google.com/file/d/1nQ08-XY4zvOLzRX-lGf8EYB5arb7-m1H/view
