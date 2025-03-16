from researchgraph.utils.openai_client import openai_client


def generate_experiment_details(model_name: str, verification_policy: str) -> str:
    prompt = f"""
Please follow the instructions below and tell us the details of the experiment for verification as described in the “Verification Policy”.
- Please answer each of the verification policies given in the “Verification Policy”.
- Please explain the details of the experiment as fully as possible. It is fine if the output is long in order to explain in detail.
- If you have any examples of experimental codes, please include them.
- Please use an experimental method that will increase the reliability of your research.
- Please make sure that the content of each experiment does not overlap too much. If several verification items can be confirmed in one experiment, please combine them into one experiment.
- Please consider the details of the experiment on the assumption that Pytorch will be used for implementation.
- Please keep in mind that you should use existing python libraries as much as possible, and avoid implementing things from scratch.
# Verification Policy
----------------------------------------
{verification_policy}
----------------------------------------"""
    output = openai_client(
        model_name=model_name,
        prompt=prompt,
    )

    return output


if __name__ == "__main__":
    # model_name = "gpt-4.5-preview-2025-02-27"
    model_name = "o3-mini-2025-01-31"
    # model_name = "o1-2024-12-17"
    verification_policy = """
Below is a concrete experimental plan featuring three realistic experiments—all implementable in Python (using libraries such as PyTorch, TensorFlow, or even NumPy for controlled settings)—to demonstrate the advantages of the Adaptive Curvature Momentum (ACM) optimizer over existing methods. ────────────────────────────── 1. Convergence and Generalization on Benchmark Datasets Objective: Test whether ACM provides faster convergence and better generalization compared to optimizers like Adam and SGD with momentum. Plan: • Choose a well-known dataset (such as CIFAR-10 or MNIST) and a common architecture (e.g., a ResNet or a simple CNN). • Train the model under the same hyperparameter regime (or a tuned regime) using ACM, Adam, and SGD. • Compare performance metrics including training loss curves, validation loss/accuracy, and the number of epochs required to reach a specific accuracy threshold. • Optionally, use learning rate schedulers or weight decay settings equally among optimizers to isolate the effect of curvature adaptation. Coding Considerations: • Implement ACM as a custom optimizer in PyTorch by extending torch.optim.Optimizer (or in TensorFlow using tf.keras.optimizers.Optimizer). • Log intermediate gradients (g_t) and use the simple difference (Δg) to scale the update in a curvature-aware manner. • Save checkpoints and plot learning curves using libraries like Matplotlib. ────────────────────────────── 2. Stability and Robustness on Synthetic Loss Landscapes Objective: Show that ACM’s curvature-aware adjustment yields improved stability in environments with varying curvature compared to traditional methods. Plan: • Construct synthetic loss functions with controlled curvature properties (e.g., quadratic bowls with different curvatures or non-convex functions with varying local curvature). • Run optimization experiments on these functions using ACM and baseline optimizers, then compare the ability of each to “navigate” flat regions versus sharp valleys. • Record convergence speed, step sizes, and the behavior near saddle points or narrow minima. Coding Considerations: • Use NumPy or PyTorch to define synthetic functions where gradients and Hessian approximations can be computed analytically or approximated easily. • Log and visualize the trajectory of the optimizer in the loss landscape. • For example, visualize the optimizer’s path against contour plots of the loss landscape to highlight curvature adaptation effects. ────────────────────────────── 3. Adaptive Step-length Analysis on a Controlled Quadratic Function Objective: Isolate and evaluate the effect of using the curvature term on update scaling by testing on a simple quadratic function whose curvature is known analytically. Plan: • Define a quadratic function f(x) = ½ x^T A x where A is a positive-definite matrix with known eigenvalues. • This function’s landscape has constant curvature properties, making it ideal for studying the behavior of adaptive step lengths. • Run experiments by initializing at a point away from the minimum and track how quickly ACM converges compared to a baseline optimizer. • Measure the adaptive learning rate adjustments and compare them with theoretical expectations given A’s eigenvalues. Coding Considerations: • Implement the quadratic function and its gradient in Python using NumPy or PyTorch. • Explicitly compute or simulate the curvature information (e.g., through the change in the gradient vector Δg) for ACM. • Plot the evolution of the adaptive learning rate (η_t) over iterations, and compare with the standard momentum update to show that larger step sizes are taken in flatter directions and smaller ones near sharper curvature. ────────────────────────────── Overall Benefits Assessed: • Faster convergence in standard deep learning settings (Experiment 1). • More stable optimization trajectories in variable curvature landscapes (Experiment 2). • Controlled, quantifiable adaptive step-sized behavior on a quadratic function that reinforces the theoretical motivation behind ACM (Experiment 3). Each of these experiments is implementable using Python-based deep learning libraries and allows not only quantitative comparisons (loss curves, convergence time) but also qualitative insights (trajectory plots, adaptive learning rate behavior). This multi-faceted approach will help demonstrate the superiority of the New ACM optimizer in both practical deep learning tasks and controlled theoretical settings.
"""
    output = generate_experiment_details(
        model_name=model_name,
        verification_policy=verification_policy,
    )
    print(output)
