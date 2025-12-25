**Computational Neuroscience Project**

In this project I am investigating the fundamental learning mechanisms in the brain, exploring alternative learning algorithms to backpropagation.

Initially, I started out exploring networks with local learning rules to solve the blind signal separation problem. Progress on this can be found in
**PartBProjectProgress.pdf**. My initial code is in the initial_experiments folder. I also explored signal separation for natural images, whose code can 
be found under the RealImageSeparation folder. My more recent exploration of VICReg and PyTorch implementations are in the PyTorchVICReg folder.

As a slightly tangential exploration, I also explored the predictive dendrites model which is similar to predictive coding models which successfully learn, 
but is even more biologically plausible. However, this model currently doesn't effectively learn - I explored training it on simple datasets and tried 
various modification ideas to see where it fails and what can be done.
