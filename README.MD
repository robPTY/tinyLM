# tinyLM

From scratch implementation of a 52M parameter Transformer, following the Vaswani et al. paper. The training will be focused on English-Spanish translation, with RL techniques performed after the model is fully trained.

Originally, I was going to code it all by hand (no autograd, no nn.Modules, etc.) as I have done with other projects, but I realized that in order to maximize my learning, I'd just use all the built-in PyTorch utils.
