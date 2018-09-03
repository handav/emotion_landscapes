This is a repo that goes along with my OpenAI project 'Generating Emotional Landscapes.' This includes two multiscale VAE architectures (one for 32x32 images and one for 64x64), a utils file, a data loader file (emo_landscapes_loader.py), and the training script (train_mscvae.py).

Multiscale conditional VAE based on the framework of Denton et al (2015). Thank you to Emily Denton for her help: https://cs.nyu.edu/~denton/

To run (sample argument parameters included):
```
python train_mscvae.py --beta 0.00005 --all_labels --z_dim 8 --image_width 64 --nlevels 4 --save_model
```

See all additional arguments in the train_mscvae.py script.

http://hannahishere.com/\
https://twitter.com/ahandvanish

References:

Denton, E., Chintala, S., Szlam, A., and Fergus, R. (2015). Deep generative image models using a laplacian pyramid of adversarial networks. In Advances in Neural Information Processing Systems (NIPS).