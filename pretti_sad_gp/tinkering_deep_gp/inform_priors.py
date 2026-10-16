import gpytorch


gp = gpytorch.models.ExactGP(X_f, f_lf, gpytorch.likelihoods.GaussianLikelihood())
f_lf_gp.covar_module = ScaleKernel(RBFKernel(ard_num_dims=X_f.shape[1]))
f_lf_gp.mean_module = ConstantMean()
f_lf_likelihood = f_lf_gp.likelihood

f_ls_map = f_lf_gp.covar_module.base_kernel.lengthscale.detach().clone()
f_os_map = f_lf_gp.covar_module.outputscale.detach().clone()
f_noise_map = f_lf_likelihood.noise.detach().clone()
f_mean_map = f_lf_gp.mean_module.constant.detach().clone()
