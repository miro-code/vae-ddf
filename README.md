# Variational AutoEncoders with Differentiable Deep Decision Forests

This is a project on semi-supervised learning using Variational AutoEncoders (VAEs). 
We implemented a differentiable random forest and a differentiable gradient boosted decision trees class in pytorch. 
To classify samples we first train a VAE with ELBO loss to extract latent representations of the training data. Then we train a downstream decision forest classifier to make predictions. 

The key idea is that we can use the distribution over the latent space to approximate gradients and tune the model using SGD. This enables further improving performance for the decision forest. It works for both Random Forest and Gradient Boosted Decision Trees.
