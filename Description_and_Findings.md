# Predicting-Daily-Temperature-using-Time-Series-Analysis
Time Series Analysis for predicting daily temperature using forecasting models such as Holt-Winters, SARMA, ARIMA, and SARIMA compared based on predictive accuracy.

Among the time series models (ARMA, ARIMA, SARIMA), the SARIMA model has the highest AIC and BIC values, suggesting that it may not be the best fit for the data. Additionally, it has a relatively higher Ljung-Box statistic (Q) indicating some potential issues with residuals and normality. 

The ARIMA model performs slightly better in terms of AIC and BIC compared to SARIMA, but it still has some issues with the Ljung-Box statistic 

The multiple linear regression model stands out with a significantly lower MSE, RMSE, and a high R-squared value of 0.8595, indicating a strong fit to the data. The adjusted R-squared value of 0.8952 suggests that the predictors in the regression model explain a substantial amount of variance. 

Considering the performance metrics, the multiple linear regression model appears to be the better model for the data. It outperforms the time series models (ARMA, ARIMA, SARIMA) in terms of predictive accuracy and goodness of fit. Therefore, the multiple linear regression model is the final model selected for this dataset.
![image](https://github.com/user-attachments/assets/95e12ce9-86a7-471a-b535-d75ef6bd5423)
