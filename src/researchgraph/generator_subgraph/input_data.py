generator_subgraph_input_data = {
    "add_method_texts": [
        '{"arxiv_id":"2312.07586v5","arxiv_url":"http://arxiv.org/abs/2312.07586v5","title":"Characteristic Guidance: Non-linear Correction for Diffusion Model at\\n  Large Guidance Scale","authors":["Candi Zheng","Yuan Lan"],"published_date":"2023-12-11T02:40:40Z","journal":"","doi":"","summary":"Popular guidance for denoising diffusion probabilistic model (DDPM) linearly\\ncombines distinct conditional models together to provide enhanced control over\\nsamples. However, this approach overlooks nonlinear effects that become\\nsignificant when guidance scale is large. To address this issue, we propose\\ncharacteristic guidance, a guidance method that provides first-principle\\nnon-linear correction for classifier-free guidance. Such correction forces the\\nguided DDPMs to respect the Fokker-Planck (FP) equation of diffusion process,\\nin a way that is training-free and compatible with existing sampling methods.\\nExperiments show that characteristic guidance enhances semantic characteristics\\nof prompts and mitigate irregularities in image generation, proving effective\\nin diverse applications ranging from simulating magnet phase transitions to\\nlatent space sampling.","github_url":"https://github.com/coderpiaobozhe/classifier-free-diffusion-guidance-Pytorch","main_contributions":"The paper introduces \'characteristic guidance\', a novel guidance method for denoising diffusion probabilistic models (DDPMs) that corrects for non-linear effects associated with large guidance scales. It demonstrates improvements in semantic alignment during image generation across various applications, effectively addressing color inconsistencies and sample diversity issues that arise with standard guidance techniques.","methodology":"Characteristic guidance corrects classifier-free guidance using a non-linear correction term derived from the Fokker-Planck equation of the diffusion process. The method of characteristics is employed to construct the correction term without requiring additional training, allowing it to integrate with existing fast sampling methods.","experimental_setup":"Experiments were conducted using several datasets including CIFAR-10 and ImageNet 256. Performance metrics included Kullback-Leibler divergence for theoretical reference comparisons, Frechet Inception Distance (FID), and Inception Score (IS) for evaluating image quality. The system was tested under different guidance scales and sampling methods such as SDE and DDIM.","limitations":"The study acknowledges that FID\'s effectiveness may be compromised due to the difference in target conditional probabilities and marginal probabilities. Furthermore, the iterative correction process required for characteristic guidance can slow down performance, and the need for effective regularization techniques is highlighted.","future_research_directions":"Future work could focus on improving the computational efficiency of the characteristic guidance through better regularization methods and exploring its applications beyond image generation, such as in other domains involving continuous data types."}',
        '{"arxiv_id":"2404.14507v1","arxiv_url":"http://arxiv.org/abs/2404.14507v1","title":"Align Your Steps: Optimizing Sampling Schedules in Diffusion Models","authors":["Amirmojtaba Sabour","Sanja Fidler","Karsten Kreis"],"published_date":"2024-04-22T18:18:41Z","journal":"","doi":"","summary":"Diffusion models (DMs) have established themselves as the state-of-the-art\\ngenerative modeling approach in the visual domain and beyond. A crucial\\ndrawback of DMs is their slow sampling speed, relying on many sequential\\nfunction evaluations through large neural networks. Sampling from DMs can be\\nseen as solving a differential equation through a discretized set of noise\\nlevels known as the sampling schedule. While past works primarily focused on\\nderiving efficient solvers, little attention has been given to finding optimal\\nsampling schedules, and the entire literature relies on hand-crafted\\nheuristics. In this work, for the first time, we propose a general and\\nprincipled approach to optimizing the sampling schedules of DMs for\\nhigh-quality outputs, called $\\\\textit{Align Your Steps}$. We leverage methods\\nfrom stochastic calculus and find optimal schedules specific to different\\nsolvers, trained DMs and datasets. We evaluate our novel approach on several\\nimage, video as well as 2D toy data synthesis benchmarks, using a variety of\\ndifferent samplers, and observe that our optimized schedules outperform\\nprevious hand-crafted schedules in almost all experiments. Our method\\ndemonstrates the untapped potential of sampling schedule optimization,\\nespecially in the few-step synthesis regime.","github_url":"https://github.com/deep-floyd/IF","main_contributions":"The paper introduces Align Your Steps (AYS), a novel framework for optimizing sampling schedules in diffusion models (DMs), addressing slow sampling speed issues while significantly improving output quality, especially in few-step syntheses.","methodology":"AYS leverages methods from stochastic calculus to optimize sampling schedules by minimizing the Kullback-Leibler divergence between true and linearized generative stochastic differential equations, tailored to specific datasets, models, and solvers.","experimental_setup":"The proposed method was evaluated on various benchmarks, including CIFAR10, FFHQ, ImageNet, and Stable Diffusion models, utilizing metrics such as FID scores and human evaluations, with a focus on a range of stochastic and deterministic samplers across different datasets.","limitations":"The approach is primarily verified for stochastic SDE solvers, which may limit its applicability across all types of solvers; also, the KLUB optimization does not always align perfectly with minimizing actual output distribution divergences, requiring careful stopping during optimization to avoid overfitting.","future_research_directions":"Future research may explore applying AYS to label- or text-conditional schedule optimization and extend its use to single-step higher-order ODE solvers as well as other generative techniques."}',
        '{"arxiv_id":"2403.03852v1","arxiv_url":"http://arxiv.org/abs/2403.03852v1","title":"Accelerating Convergence of Score-Based Diffusion Models, Provably","authors":["Gen Li","Yu Huang","Timofey Efimov","Yuting Wei","Yuejie Chi","Yuxin Chen"],"published_date":"2024-03-06T17:02:39Z","journal":"","doi":"","summary":"Score-based diffusion models, while achieving remarkable empirical\\nperformance, often suffer from low sampling speed, due to extensive function\\nevaluations needed during the sampling phase. Despite a flurry of recent\\nactivities towards speeding up diffusion generative modeling in practice,\\ntheoretical underpinnings for acceleration techniques remain severely limited.\\nIn this paper, we design novel training-free algorithms to accelerate popular\\ndeterministic (i.e., DDIM) and stochastic (i.e., DDPM) samplers. Our\\naccelerated deterministic sampler converges at a rate $O(1/{T}^2)$ with $T$ the\\nnumber of steps, improving upon the $O(1/T)$ rate for the DDIM sampler; and our\\naccelerated stochastic sampler converges at a rate $O(1/T)$, outperforming the\\nrate $O(1/\\\\sqrt{T})$ for the DDPM sampler. The design of our algorithms\\nleverages insights from higher-order approximation, and shares similar\\nintuitions as popular high-order ODE solvers like the DPM-Solver-2. Our theory\\naccommodates $\\\\ell_2$-accurate score estimates, and does not require\\nlog-concavity or smoothness on the target distribution.","github_url":"https://github.com/huggingface/diffusers","main_contributions":"This paper addresses the speed limitations of score-based diffusion models during sampling, proposing novel training-free algorithms that improve the convergence rates of both deterministic (DDIM) and stochastic (DDPM) samplers, achieving rates of O(1/T^2) and O(1/T) respectively.","methodology":"The algorithms utilize insights from higher-order approximations and are based on the ODE and SDE frameworks of diffusion models, leveraging momentum terms in the update rules to enhance sampling speed without extensive retraining.","experimental_setup":"The performance of the proposed samplers was validated using pre-trained score functions from three datasets: CelebA-HQ, LSUN-Bedroom, and LSUN-Churches. Evaluations focused on image quality generated using different numbers of function evaluations (NFEs).","limitations":"The theoretical guarantees may have sub-optimal dependencies on the problem dimension, and the convergence results assume ℓ2-accurate score estimates without needing log-concavity or smoothness on target distributions.","future_research_directions":"Future work could explore refining the theory to sharpen dimension dependencies, and developing higher-order solvers for SDE-based samplers, building on connections to third-order ODE approximations."}',
    ],
    "base_method_text": """\
'{"arxiv_id":"2402.02149v2","arxiv_url":"http://arxiv.org/abs/2402.02149v2","title":"Improving Diffusion Models for Inverse Problems Using Optimal Posterior\\n  Covariance","authors":["Xinyu Peng","Ziyang Zheng","Wenrui Dai","Nuoqian Xiao","Chenglin Li","Junni Zou","Hongkai Xiong"],"published_date":"2024-02-03T13:35:39Z","journal":"","doi":"","summary":"Recent diffusion models provide a promising zero-shot solution to noisy\\nlinear inverse problems without retraining for specific inverse problems. In\\nthis paper, we reveal that recent methods can be uniformly interpreted as\\nemploying a Gaussian approximation with hand-crafted isotropic covariance for\\nthe intractable denoising posterior to approximate the conditional posterior\\nmean. Inspired by this finding, we propose to improve recent methods by using\\nmore principled covariance determined by maximum likelihood estimation. To\\nachieve posterior covariance optimization without retraining, we provide\\ngeneral plug-and-play solutions based on two approaches specifically designed\\nfor leveraging pre-trained models with and without reverse covariance. We\\nfurther propose a scalable method for learning posterior covariance prediction\\nbased on representation with orthonormal basis. Experimental results\\ndemonstrate that the proposed methods significantly enhance reconstruction\\nperformance without requiring hyperparameter tuning.","github_url":"https://github.com/xypeng9903/k-diffusion-inverse-problems","main_contributions":"The paper addresses the problem of improving diffusion models for solving noisy linear inverse problems by optimizing the posterior covariance without retraining. The key findings include the development of plug-and-play solutions for posterior covariance that significantly enhance reconstruction performance across various tasks such as inpainting, deblurring, and super-resolution without requiring hyperparameter tuning.","methodology":"The proposed approach replaces hand-crafted isotropic covariance approximations with more principled covariance determined by maximum likelihood estimation. Methods include a generalized optimization for posterior covariance using reverse covariance prediction or Monte Carlo estimations and a scalable method based on orthonormal basis transformations to learn posterior covariance predictions.","experimental_setup":"Experiments were conducted on FFHQ and ImageNet datasets to validate the proposed methods against existing ones like DPS and ΠGDM. Performance metrics included SSIM, LPIPS, and FID across tasks such as image inpainting (50% masking), Gaussian and motion deblurring, and 4× super-resolution. The experimental setups ensured consistent evaluation by using a unified codebase for implementation.","limitations":"The proposed methods, constrained by diagonal covariance assumptions, may not achieve optimal performance even with perfect model training. Additional challenges include potential estimation errors related to the reverse covariance predictions and limitations in addressing high-dimensional data complexities.","future_research_directions":"Future work could investigate the design of better covariance principles, explore nonlinear transformations for improved correlation reduction, and develop efficient approximation methods similar to Tweedie\'s approach for broader application in inverse problems."}'"
""",
}
