# Core Assumption of Naive Bayes
#  It assumes feature independence given the
# class label—that is, the presence or absence
#  of a feature doesn’t influence any other feature within 
#  the same class. This “naive” assumption simplifies probability
#   calculations using Bayes' theorem.


# GaussianNB vs. MultinomialNB vs. BernoulliNB
# These are variants tailored to different data types:
# GaussianNB: Best for continuous features, assuming they follow a normal distribution.
# MultinomialNB: Designed for count data (e.g. word frequencies in text classification).
# BernoulliNB: Suited for binary/Boolean features, where each feature is either present (1) or absent (0).


# Naive Bayes & High-Dimensional Data
#  Since Naive Bayes calculates probabilities independently for each feature, 
# it avoids the curse of dimensionality. It’s computationally efficient
# and performs well in high-dimensional spaces—like text classification—where 
# other models might struggle.