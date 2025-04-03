from researchgraph.utils.openai_client import openai_client


def generate_experiment_code(
    llm_name: str, experiment_details: str, experiment_info_of_source_research: str
) -> str:
    prompt = f"""
# Introduction
Please follow the instructions below to tell us the detailed code for conducting the experiment.
- Please output the detailed experiment code for each experiment.
- As you will be checking the results of the experiment from the standard output, please include print statements, etc. in your implementation so that the contents of the experiment and its results, etc. can be accurately understood from the standard output.
- Please add a function to test the code to check that it is executed correctly. As the test is to check that the code is working correctly, please make it so that the test finishes immediately.
- Please implement all frameworks used for deep learning in pytorch.
- When conducting experiments, please prepare multiple patterns of data and create an experimental code that demonstrates the robustness of the new method.
- Please also output the names of the python libraries that you think are necessary for running the experiment.
- The section 'Experimental information from the research on which it is based' includes details about the experiments conducted in the original research. Please use this information to implement the experiment as closely as possible to the original.
- Please use matplotlib or seaborn to plot the results (e.g., accuracy, loss curves, confusion matrix), 
and **explicitly save all plots as `.pdf` files using `plt.savefig("filename.pdf")` or equivalent.
    - Do not use `.png` or other formats—output must be `.pdf` only. These plots should be suitable for inclusion in academic papers.
- Use the following filename format:
    <figure_topic>[_<condition>][_pairN].pdf
    - `<figure_topic>`: the main subject of the figure (e.g., `training_loss`, `accuracy`, `inference_latency`)
    - `_<condition>`(optional): a specific model, setting, or comparison (e.g., `amict`, `baseline`, `tokens`, `multimodal_vs_text`)
    - `_pairN`(optional): indicates that the figure is part of a pair (e.g., `_pair1`, `_pair2`) to be shown side by side using subfigures
    

# Experiment Details
{experiment_details}
# Experimental information from the research on which it is based
{experiment_info_of_source_research}"""
    messages = [
        {"role": "user", "content": prompt},
    ]
    response = openai_client(model_name=llm_name, message=messages)
    return response


if __name__ == "__main__":
    # model_name = "gpt-4.5-preview-2025-02-27"
    model_name = "o3-mini-2025-01-31"
    # model_name = "o1-2024-12-17"
    experiment_details = """
Below is a comprehensive experimental plan that addresses each part of the Verification Policy. The plan details three experiments that (a) compare convergence and generalization on a standard benchmark, (b) study stability on synthetic loss landscapes, and (c) isolate the adaptive step‐length behavior on a simple quadratic function. In all experiments we assume a PyTorch implementation and use standard libraries (such as Matplotlib for plotting, NumPy for simple computations, etc.). For clarity, each experiment starts with an explanation of the objectives, followed by the experimental design, coding considerations (with example code snippets), and then a discussion on the expected outcomes. ────────────────────────────── 1. Convergence and Generalization on Benchmark Datasets Objective: • Compare the new Adaptive Curvature Momentum (ACM) optimizer with popular alternatives such as Adam and SGD with momentum. • Demonstrate faster convergence (lower training loss earlier) and/or better generalization (higher validation accuracy) on standard datasets (e.g., CIFAR-10 or MNIST). Experimental Plan: • Dataset and Architecture: – Use CIFAR-10 as the benchmark dataset. – Select a common convolutional neural network (CNN) architecture (for example, a simple 4–layer CNN or a light ResNet variant). • Optimizers Setup: – Implement ACM as a custom optimizer in PyTorch (see “Custom ACM Optimizer Code” below). – Train the same network using ACM, Adam, and SGD (with momentum) under identical or carefully tuned hyperparameters. – Employ the same learning rate scheduling and weight decay settings across optimizers so that any differences can be mainly attributed to curvature adaptation. • Evaluation: – Log training loss curves, validation loss/accuracy, and the number of epochs required to reach a specific accuracy threshold. – Save model checkpoints and record the evolution of gradients (g_t) and differences (Δg) to verify the curvature adaptation in ACM. • Analysis: – Plot learning curves side by side using Matplotlib. – Identify whether ACM provides a steeper initial descent or improved final accuracy. Example Code Snippet for ACM Optimizer: ---------------------------------------------------------------- import torch from torch.optim.optimizer import Optimizer, required class ACM(Optimizer): def __init__(self, params, lr=required, momentum=0.0): defaults = dict(lr=lr, momentum=momentum) super(ACM, self).__init__(params, defaults) # Dictionary to store previous gradients for each parameter self.prev_grads = {} def step(self, closure=None): loss = None if closure is not None: loss = closure() # Loop over parameter groups for group in self.param_groups: lr = group['lr'] momentum = group['momentum'] for p in group['params']: if p.grad is None: continue grad = p.grad.data.clone() # Retrieve previous gradient; initialize if missing if p not in self.prev_grads: self.prev_grads[p] = grad.clone() delta_grad = grad - self.prev_grads[p] # Estimate curvature: using the norm of delta_grad. # In practice, one might adapt this estimate. curvature_estimate = 1.0 / (delta_grad.norm() + 1e-8) # Adapt learning rate using curvature adaptive_lr = lr * curvature_estimate # Apply momentum if available state = self.state[p] if 'velocity' not in state: state['velocity'] = torch.zeros_like(p.data) velocity = state['velocity'] # Update rule: velocity is updated using the adaptive learning rate velocity.mul_(momentum).add_(grad, alpha=adaptive_lr) p.data.add_(-velocity) # Update the stored previous gradient self.prev_grads[p] = grad.clone() return loss # Example usage in training: # optimizer = ACM(model.parameters(), lr=0.01, momentum=0.9) ---------------------------------------------------------------- Training Procedure Example: • Set up training loops for each optimizer. • Use common libraries like torch.utils.data.DataLoader for data handling and Matplotlib for plotting loss/accuracy curves. • Optionally, log additional information (like gradient norms) to a file for further analysis. ────────────────────────────── 2. Stability and Robustness on Synthetic Loss Landscapes Objective: • Examine how curvature‐aware scaling in ACM influences the optimizer’s behavior on loss landscapes with variable curvature. • Compare trajectories of ACM against classical optimizers when navigating flat regions, steep valleys, and near saddle points. Experimental Plan: • Synthetic Functions: – Create synthetic loss landscapes in 2D that mimic varying curvature. For example: □ A quadratic bowl with different scaling along each coordinate: f(x, y) = 0.5*(a*x² + b*y²) with a ≠ b, which creates “elongated” contours. □ Non-convex functions with multiple local minima and saddle points, e.g., a modified Rosenbrock function. • Experiment Setup: – Use PyTorch tensors with requires_grad=True to represent the optimization variables. – Initialize the optimizer (ACM, Adam, SGD) at the same starting point. – (Optionally) Log the parameter updates, step sizes, and adaptive learning rates. • Visualization: – Plot contours of the synthetic loss function using Matplotlib. – Overlay the path taken by each optimizer to visually compare how they handle regions with different curvature. • Analysis: – Compare convergence speed and trajectory smoothing. – Discuss the ability to maintain stability in flat regions vs. correct stepping in steep areas. Example Code Snippet for a Synthetic Quadratic Function: ---------------------------------------------------------------- import torch import numpy as np import matplotlib.pyplot as plt # Define a quadratic function with anisotropic curvature def quadratic_loss(z, a=1.0, b=10.0): # z is a tensor with shape (2,) x, y = z[0], z[1] loss = 0.5 * (a * x**2 + b * y**2) return loss # Setting up a grid for contour plotting x_vals = np.linspace(-3, 3, 100) y_vals = np.linspace(-3, 3, 100) X, Y = np.meshgrid(x_vals, y_vals) Z = 0.5 * (1 * X**2 + 10 * Y**2) # Plot contours plt.figure(figsize=(6, 5)) contours = plt.contour(X, Y, Z, levels=30, cmap='viridis') plt.clabel(contours, inline=True, fontsize=8) plt.title("Quadratic Loss Landscape with Anisotropic Curvature") plt.xlabel("x") plt.ylabel("y") # Example: tracking optimization using ACM on this function # Starting point z = torch.tensor([3.0, 3.0], requires_grad=True) # Instantiate ACM optimizer for a single parameter vector optimizer_acm = ACM([z], lr=0.1, momentum=0.9) trajectory = [] # Optimization loop (for example, 50 iterations) for i in range(50): optimizer_acm.zero_grad() loss = quadratic_loss(z) loss.backward() optimizer_acm.step() trajectory.append(z.detach().clone().numpy()) trajectory = np.array(trajectory) plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color='red', label='ACM trajectory') plt.legend() plt.show() ---------------------------------------------------------------- Interpretation: • Look at how the red trajectory (ACM’s path) compares to expected “ideal” paths that move quickly in the flat (low curvature) direction and carefully in the steep direction. • Repeat the same procedure for Adam and SGD (plotting their trajectories with different colors) to illustrate the differences. ────────────────────────────── 3. Adaptive Step-length Analysis on a Controlled Quadratic Function Objective: • Isolate the effect of the curvature term on update scaling by testing on a quadratic function whose curvature is known analytically. • Examine whether ACM’s adaptive learning rate adjustments conform to theoretical expectations when A’s eigenvalues are controlled. Experimental Plan: • Define a quadratic function: f(x) = ½ x^T A x, where A is a positive-definite matrix. – For example, choose a 2D function with A = diag(λ1, λ2) such that λ1 ≠ λ2. • Experimental Setup: – Initialize the variable away from the optimum. – Run the optimization for a fixed number of iterations with ACM and, as a baseline, an optimizer that does not adapt the step length (e.g., SGD with momentum). – At each iteration, log the adaptive learning rate (η_t) computed by the ACM algorithm. • Analysis: – Plot the evolution of η_t over iterations. – Compare the step sizes taken in directions corresponding to λ1 versus λ2. – Validate that larger steps are made in the flatter directions and smaller steps in sharper directions. Example Code Snippet for Controlled Quadratic Function: ---------------------------------------------------------------- import torch import matplotlib.pyplot as plt # Define a 2D quadratic function with known curvature eigenvalues. # Let A = diag(lambda1, lambda2) lambda1, lambda2 = 0.5, 20.0  # e.g., one flat and one steep direction def quadratic_function(z): return 0.5 * (lambda1 * z[0]**2 + lambda2 * z[1]**2) # Create a parameter vector to optimize z = torch.tensor([5.0, 5.0], requires_grad=True) # Use our ACM optimizer (modified to store adaptive lr history for this experiment) class ACMwithHistory(ACM): def __init__(self, params, lr=required, momentum=0.0): super().__init__(params, lr, momentum) self.adaptive_lr_history = [] def step(self, closure=None): # We average the adaptive lr across parameters for monitoring purposes adaptive_lrs = [] for group in self.param_groups: lr = group['lr'] momentum = group['momentum'] for p in group['params']: if p.grad is None: continue grad = p.grad.data.clone() if p not in self.prev_grads: self.prev_grads[p] = grad.clone() delta_grad = grad - self.prev_grads[p] curvature_estimate = 1.0 / (delta_grad.norm() + 1e-8) adaptive_lr = lr * curvature_estimate adaptive_lrs.append(adaptive_lr) state = self.state[p] if 'velocity' not in state: state['velocity'] = torch.zeros_like(p.data) velocity = state['velocity'] velocity.mul_(momentum).add_(grad, alpha=adaptive_lr) p.data.add_(-velocity) self.prev_grads[p] = grad.clone() # Record the mean adaptive lr this step if adaptive_lrs: self.adaptive_lr_history.append(sum(adaptive_lrs) / len(adaptive_lrs)) return None optimizer_acm = ACMwithHistory([z], lr=0.1, momentum=0.9) adaptive_lr_record = [] loss_record = [] # Optimization loop for a fixed number of steps, e.g., 50 iterations for i in range(50): optimizer_acm.zero_grad() loss = quadratic_function(z) loss.backward() optimizer_acm.step() loss_record.append(loss.item()) adaptive_lr_record = optimizer_acm.adaptive_lr_history # Plotting the adaptive lr evolution and loss decay plt.figure(figsize=(10, 4)) plt.subplot(1, 2, 1) plt.plot(adaptive_lr_record, marker='o') plt.title("Adaptive Learning Rate over Iterations") plt.xlabel("Iteration") plt.ylabel("Adaptive lr") plt.subplot(1, 2, 2) plt.plot(loss_record, marker='o', color='purple') plt.title("Loss Value over Iterations") plt.xlabel("Iteration") plt.ylabel("Loss") plt.tight_layout() plt.show() ---------------------------------------------------------------- Interpretation: • The adaptive learning rate plot should show that ACM reduces the step size when the difference in gradients (and thus local curvature) is large. • By comparing these recorded values to the eigenvalues of A, one verifies that directions corresponding to higher curvature (larger eigenvalue) receive smaller adaptive steps. • The loss curve will also highlight faster convergence relative to a non-adaptive baseline (if such a run is conducted in parallel). ────────────────────────────── Overall Reliability and Reproducibility To ensure your research is as reliable as possible: • Use multiple runs (e.g., 5–10 random seeds) to average performance metrics and provide error bars where relevant. • Combine quantitative metrics (loss/accuracy curves, convergence epoch count) with qualitative visualizations (trajectory plots, adaptive lr evolution). • Use stable and well-tested libraries (PyTorch, NumPy, Matplotlib) to reduce bugs. • Publish complete code and configuration details (hyperparameters, random seeds, version numbers) to enable reproducibility. Each experiment described here is distinct yet complementary: • Experiment 1 validates performance in real-world settings. • Experiment 2 demonstrates improved stability across varying curvature landscapes. • Experiment 3 quantifies the adaptive mechanism on a well-defined quadratic function. By executing these experiments and comparing ACM with standard optimizers side-by-side, you can robustly verify the advantages of the Adaptive Curvature Momentum optimizer. This detailed plan, along with the provided code examples, should serve as a strong foundation for verifying the claimed benefits of ACM through multiple, non-overlapping experimental setups.
"""
    output = generate_experiment_code(
        llm_name=model_name,
        experiment_details=experiment_details,
    )
    print(output)
