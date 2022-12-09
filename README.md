# CS105 Final Project - Creating a Simple Neural Network

Members:
* Jackson Hoke
* Jason Houang
* Ayush Nabar
* Nathan Ng
* Sumachai Suksanguan

## Introduction

As we get busier and busier, we have less time to decide if things are good.  Movies in particular thrive from word-of-mouth from others and critic reviews.  Strongly worded reviews help distringuish a movie and give viewers something to look forward to.  We are building a simple neural network utilizing a bag-of-words model to create the necessary features from a collection of Amazon reviews.  The neural network will use each word in the review as features, while each row is a separate sentence to predict movie review star ratings.

## Goal

To predict the star ratings (0 - 5) of a review.

## Data
Amazon product data obtained from [here](jmcauley.ucsd.edu/data/amazon).  The data used was reviews from Amazon Instand Video, which contained information such as `reviewerID`, `asin`, `reviewerName` and others, but for the sake of this project we are focusing mainly on:
* `reviewText`: Sentence-form review of the video. 
* `overall`: Rating of the review in integer star format.

## Tools Used

* Ask
* Nathan
* and
* Jackson 

## Pre-processing

In order to even create such a network, we need to clean the data to look solely at those words.

```py
reviews = pd.read_json('reviews_Amazon_Instant_Video_5.json', lines=true)       # Read in Our Data

vectorizer = CountVectorizer(stop_words='english', min_df=100)                  # Uses Stop Words to remove filler "English Words"

cdf = vectorizer.fit_transform(reviews['reviewText'])                           # Runs vectorizer on our review text
```

Using these above functions allows us to read in our data, then process our data with `CountVectorizer`.  `CountVectorizer` takes in multiple parameters to help with further cleaning the data including:
* `stop_words='english'`: Removes all instances of known-english stop words (i.e, Filler Words)
* `min_df=100`: Ignores terms that have a frequency strictly lower than our given threshold, 100.

After we run initial cleaning, we can use the vectorizer to create our Bag-Of-Words model:
```py
bag = pd.DataFrame(cdf.toarray(), columns=vectorizer.get_feature_names_out())
```

Initial Bag-of-Words Model will look like this:

![Bag_Words](./report_images/initial_data.png)

## Data Analysis and Manipulation (Jackson and Nathan)

Now it's time to actually create our neural network.  For the sake of this assignment, the scope of the review will be defined as such:
* Review of __**4 or higher**__ being a __positive__ review
* Review of __**3 or lower**__ being a __negative__ review

```py
x = bag.values                  # Our Bag-of-Words model
y = reviews['overall'].values   # User Rating for the Film they are reviewing

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

Use the given `test_train_splits()` function to create our the basis of our neural networking, designating these particular parameters:
* `test_size=0.2`: 20% of the data will be testing
* `random_state=42`: Seed value so that we can replicate our results in testing

Now, we can actually create our neural network!  A common tactic is to utilize a **Multi-Layered Perceptron (MLP)** to make our classifications. 

```py
mlp = MLPRegressor(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
```

The MLP Regressor takes these parameters in order to create our network:
* `hidden_layer_sizes=(8,8,8)`: Represents the number of neurons in the ith hidden layer
* `activation='relu'`: Uses `relu` as the activation function, returning `f(x) = max(0,x)` by using a rectified linear unit function
* `solver='adam'`: The default solver, which uses a stochastic gradient-based optimizer to do the weight optimization
* `max_iter=500`: Max number of iterations, which we set for 500

We can then begin to train our dataset by using the following functions:

```py
mlp.fit(x_train, y_train)

predict_train = mlp.predict(x_train)
predict_test = mlp.predict(x_test)
```

After training the data, we then proceed to see how well it does compared to it's training data that we designated earlier, `x_train` and `x_test`.

We can then proceed to calculating our **Mean Absolute Error (MAE)** and **Mean Squared Error (MSE)**

```py
print('Mean absolute error: ', sum(abs(y_train - predict_train)) / len(y_train))

# Mean absolute error: 0.28293546160037275

print('Mean squared error: ', sum((y_train - predict_train) * (y_train - predict_train)) / len(y_train))

# Mean squared error: 0.16882949596856742

print('Mean absolute error: ', sum(abs(y_test - predict_test)) / len(y_train))

# Mean absolute error: 0.2101876842196558

print('Mean squared error: ', sum((y_test - predict_test) * (y_test - predict_test)) / len(y_train))

# Mean squared error: 0.37544200766226
```

Our results yield as follows:

|         | MAE     | MSE    |
| ------- | ------- | ------ |
| y_train | 0.2829  | 0.1688 |
| y_test  | 0.2102  | 0.3754 |


## Conclusion

## Questions

1. How can neural networks be used to analyze and understand the sentiment expressed in Amazon Video reviews?
2. What preprocessing steps were necessary to prepare our dataset for use in a neural network?
3. Can a neural network be trained to accurately classify the sentiment of Amazon Video Reviews?  If so, how can this be evaluated?  For example, some reviews may seem negative, but gave a positive review.