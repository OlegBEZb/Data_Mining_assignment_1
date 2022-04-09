# Data_Mining_assignment_1

# TODOs
1. EDA
2. Cleaning
3. Decide regarding the aggregation
4. Investigate previous usages of this data
5. Find the detailed documentation (how the data was collected, who are the participants)
6. Feature engineering (day of the week and so on, hour, daytime (morning, evening), timedeltas, proxy locations (with time shifts), downtime feature (for sleeping), feature for not doing anything)
7. Modeling (Prophet, microsoftarchive/SilverlightToolkit)

## Literature
* [Dataset and pilot study. (Asselbergs et al., 2016)](https://www.jmir.org/2016/3/e72/)
* [Some VU paper that was published](https://www.researchgate.net/profile/Joost-Asselbergs/publication/303790988_Exploring_and_Comparing_Machine_Learning_Approaches_for_Predicting_Mood_Over_Time/links/5a02198c4585155c96cb8db1/Exploring-and-Comparing-Machine-Learning-Approaches-for-Predicting-Mood-Over-Time.pdf)
    * They did a `weekday` attribute.
    * Missing values:
        * If the value for mood is missing for an observation, they dropped it.
        * If the value for any other variable is missing for an observation, they imputed it with the mean of that variable for that specific participant.
    * They did 3 methods for predicting `mood`:
        * Only considering `mood` to predict `mood` (Auto Regressive Integrated Moving Average). This did not work at all. MSE: 0.475
        * Similarity analysis with Dynamic Time Warping: Basically understanding if some different individuals exhibit similar mood patterns and using that to predict for highly similar individuals. This did not end up working well either because of limited participants and great variation in similarity.
        * Full predictive modeling: They used SVM and RF to predict `mood` using the other attributes. This methodology with the added features that they calulated worked best out of the ones they tried, but the best MSE that they achieve is **0.410** for the **SVM with added features**.
